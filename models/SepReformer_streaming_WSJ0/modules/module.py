import sys
import time
sys.path.append('../')

import torch
import warnings
warnings.filterwarnings('ignore')

from utils.decorators import *
from .streaming_network import *


class AudioEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int, bias: bool):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, groups=groups, bias=bias)
        self.gelu = torch.nn.GELU()
    
    def forward(self, x: torch.Tensor):
        x = torch.unsqueeze(x, dim=0) if len(x.shape) == 1 else torch.unsqueeze(x, dim=1) # [T] - >[1, T] OR [B, T] -> [B, 1, T]
        x = self.conv1d(x)
        x = self.gelu(x)
        return x
    
class FeatureProjector(torch.nn.Module):
    def __init__(self, num_channels: int, in_channels: int, out_channels: int, kernel_size: int, bias: bool):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_groups=1, num_channels=num_channels, eps=1e-8)
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias)
    
    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.conv1d(x)
        return x


class Separator(torch.nn.Module):
    def __init__(self, num_stages: int, relative_positional_encoding: dict, enc_stage: dict, spk_split_stage: dict,
                 simple_fusion: dict, dec_stage: dict):
        super().__init__()

        class RelativePositionalEncoding(torch.nn.Module):
            def __init__(self, in_channels: int, num_heads: int, maxlen: int, embed_v=False):
                super().__init__()
                self.in_channels = in_channels
                self.num_heads = num_heads
                self.embedding_dim = self.in_channels // self.num_heads
                self.maxlen = maxlen
                self.pe_k = torch.nn.Embedding(num_embeddings=2 * maxlen, embedding_dim=self.embedding_dim)
                self.pe_v = torch.nn.Embedding(num_embeddings=2 * maxlen,
                                               embedding_dim=self.embedding_dim) if embed_v else None

            def forward(self, pos_seq: torch.Tensor):
                pos_seq.clamp_(-self.maxlen, self.maxlen - 1)
                pos_seq += self.maxlen
                pe_k_output = self.pe_k(pos_seq)
                pe_v_output = self.pe_v(pos_seq) if self.pe_v is not None else None
                return pe_k_output, pe_v_output

        class SepEncStage(torch.nn.Module):
            def __init__(self, global_blocks: dict, local_blocks: dict, down_conv_layer: dict, down_conv=True):
                super().__init__()

                class DownConvLayer(torch.nn.Module):
                    def __init__(self, in_channels: int, samp_kernel_size: int):
                        super().__init__()
                        self.down_conv = torch.nn.Conv1d(
                            in_channels=in_channels, out_channels=in_channels, kernel_size=samp_kernel_size, stride=2,
                            padding=(samp_kernel_size - 1) // 2, groups=in_channels)
                        self.BN = torch.nn.BatchNorm1d(num_features=in_channels)
                        self.gelu = torch.nn.GELU()

                    def forward(self, x: torch.Tensor):
                        x = x.permute([0, 2, 1])
                        x = self.down_conv(x)
                        x = self.BN(x)
                        x = self.gelu(x)
                        x = x.permute([0, 2, 1])
                        return x

                self.g_block_1 = GlobalBlock(**global_blocks)
                self.l_block_1 = LocalBlock(**local_blocks)
                # self.g_block_2 = GlobalBlock(**global_blocks)
                # self.l_block_2 = LocalBlock(**local_blocks)
                self.downconv = DownConvLayer(**down_conv_layer) if down_conv else None

            def forward(self, x: torch.Tensor):
                x = self.g_block_1(x)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_1(x)
                x = x.permute(0, 2, 1).contiguous()
                # x = self.g_block_2(x)
                # x = x.permute(0, 2, 1).contiguous()
                # x = self.l_block_2(x)
                # x = x.permute(0, 2, 1).contiguous()
                skip = x
                if self.downconv:
                    x = x.permute(0, 2, 1).contiguous()
                    x = self.downconv(x)
                    x = x.permute(0, 2, 1).contiguous()
                return x, skip

        class SpkSplitStage(torch.nn.Module):
            def __init__(self, in_channels: int, num_spks: int):
                super().__init__()
                self.linear = torch.nn.Sequential(
                    torch.nn.Conv1d(in_channels, 4 * in_channels * num_spks, kernel_size=1),
                    torch.nn.GLU(dim=-2),
                    torch.nn.Conv1d(2 * in_channels * num_spks, in_channels * num_spks, kernel_size=1))
                self.norm = torch.nn.GroupNorm(1, in_channels, eps=1e-8)
                self.num_spks = num_spks

            def forward(self, x: torch.Tensor):
                x = self.linear(x)
                B, _, T = x.shape
                x = x.view(B * self.num_spks, -1, T).contiguous()
                x = self.norm(x)
                return x

        class SepDecStage(torch.nn.Module):
            def __init__(self, num_spks: int, global_blocks: dict, local_blocks: dict, spk_attention: dict):
                super().__init__()
                self.g_block_1 = GlobalBlock(**global_blocks)
                self.l_block_1 = LocalBlock(**local_blocks)
                self.spk_attn_1 = SpkAttention(**spk_attention)
                # self.g_block_2 = GlobalBlock(**global_blocks)
                # self.l_block_2 = LocalBlock(**local_blocks)
                # self.spk_attn_2 = SpkAttention(**spk_attention)
                # self.g_block_3 = GlobalBlock(**global_blocks)
                # self.l_block_3 = LocalBlock(**local_blocks)
                # self.spk_attn_3 = SpkAttention(**spk_attention)
                self.num_spk = num_spks

            def forward(self, x: torch.Tensor):
                x = self.g_block_1(x)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_1(x)
                x = x.permute(0, 2, 1).contiguous()
                x = self.spk_attn_1(x, self.num_spk)
                # x = self.g_block_2(x)
                # x = x.permute(0, 2, 1).contiguous()
                # x = self.l_block_2(x)
                # x = x.permute(0, 2, 1).contiguous()
                # x = self.spk_attn_2(x, self.num_spk)
                # x = self.g_block_3(x)
                # x = x.permute(0, 2, 1).contiguous()
                # x = self.l_block_3(x)
                # x = x.permute(0, 2, 1).contiguous()
                # x = self.spk_attn_3(x, self.num_spk)
                skip = x
                return x, skip

        self.num_stages = num_stages
        self.pos_emb = RelativePositionalEncoding(**relative_positional_encoding)
        self.enc_stages = torch.nn.ModuleList([SepEncStage(**enc_stage, down_conv=True) for _ in range(num_stages)])
        self.bottleneck_G = SepEncStage(**enc_stage, down_conv=False)
        self.spk_split_block = SpkSplitStage(**spk_split_stage)
        self.simple_fusion = torch.nn.ModuleList([torch.nn.Conv1d(in_channels=simple_fusion['out_channels'] * 2,
                                                                  out_channels=simple_fusion['out_channels'],
                                                                  kernel_size=1) for _ in range(num_stages)])
        self.dec_stages = torch.nn.ModuleList([SepDecStage(**dec_stage) for _ in range(num_stages)])

    def forward(self, input: torch.Tensor):
        len_x = input.shape[-1]
        pos_seq = torch.arange(0, len_x // 2 ** self.num_stages).long().to(input.device)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]
        pos_k, _ = self.pos_emb(pos_seq)
        skip = []
        for idx in range(self.num_stages):
            on_test_start = time.time()
            x, skip_ = self.enc_stages[idx](input)
            cost_time = time.time() - on_test_start
            print("enc_stages:", cost_time)

            on_test_start = time.time()
            skip_ = self.spk_split_block(skip_)
            cost_time = time.time() - on_test_start
            print("spk_split_block:", cost_time)
            skip.append(skip_)
        on_test_start = time.time()
        x, _ = self.bottleneck_G(x)
        cost_time = time.time() - on_test_start
        print("bottleneck_G:", cost_time)
        x = self.spk_split_block(x)

        each_stage_outputs = []
        for idx in range(self.num_stages):
            each_stage_outputs.append(x)
            idx_en = self.num_stages - (idx + 1)
            x = torch.nn.functional.upsample(x, skip[idx_en].shape[-1])
            x = torch.cat([x, skip[idx_en]], dim=1)
            x = self.simple_fusion[idx](x)

            on_test_start = time.time()
            x, _ = self.dec_stages[idx](x)
            cost_time = time.time() - on_test_start
            print("dec:", cost_time)

        last_stage_output = x
        return last_stage_output, each_stage_outputs

class OutputLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_spks: int, masking: bool = False):
        super().__init__()
        self.masking = masking
        self.spe_block = Masking(in_channels, Activation_mask="ReLU", concat_opt=None)
        self.num_spks = num_spks
        self.end_conv1x1 = torch.nn.Sequential(
            torch.nn.Linear(out_channels, 4 * out_channels),
            torch.nn.GLU(),
            torch.nn.Linear(2 * out_channels, in_channels))

    def forward(self, x: torch.Tensor, input: torch.Tensor):
        x = x[..., :input.shape[-1]]
        x = x.permute([0, 2, 1])
        x = self.end_conv1x1(x)
        x = x.permute([0, 2, 1])
        B, N, L = x.shape
        B = B // self.num_spks
        if self.masking:
            input = input.expand(self.num_spks, B, N, L).transpose(0, 1).contiguous()
            input = input.view(B * self.num_spks, N, L)
            x = self.spe_block(x, input)
        x = x.view(B, self.num_spks, N, L)
        x = x.transpose(0, 1)
        return x


class AudioDecoder(torch.nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accepts 3/4D tensor as input".format(self.__class__.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        x = torch.squeeze(x, dim=1) if torch.squeeze(x).dim() == 1 else torch.squeeze(x)
        return x