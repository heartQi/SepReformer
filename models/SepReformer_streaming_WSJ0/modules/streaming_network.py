import torch
import math
import numpy
from utils.decorators import *


class LayerScale(torch.nn.Module):
    def __init__(self, dims, input_size, Layer_scale_init=1.0e-5):
        super().__init__()
        if dims == 1:
            self.layer_scale = torch.nn.Parameter(torch.ones(input_size)*Layer_scale_init, requires_grad=True)
        elif dims == 2:
            self.layer_scale = torch.nn.Parameter(torch.ones(1,input_size)*Layer_scale_init, requires_grad=True)
        elif dims == 3:
            self.layer_scale = torch.nn.Parameter(torch.ones(1,1,input_size)*Layer_scale_init, requires_grad=True)
    
    def forward(self, x):
        return x*self.layer_scale

class Masking(torch.nn.Module):
    def __init__(self, input_dim, Activation_mask='Sigmoid', **options):
        super(Masking, self).__init__()
        
        self.options = options
        if self.options['concat_opt']:
            self.pw_conv = torch.nn.Conv1d(input_dim*2, input_dim, 1, stride=1, padding=0)

        if Activation_mask == 'Sigmoid':
            self.gate_act = torch.nn.Sigmoid()
        elif Activation_mask == 'ReLU':
            self.gate_act = torch.nn.ReLU()
            

    def forward(self, x, skip):
   
        if self.options['concat_opt']:
            y = torch.cat([x, skip], dim=-2)
            y = self.pw_conv(y)
        else:
            y = x
        y = self.gate_act(y) * skip

        return y


class GCFN(torch.nn.Module):
    def __init__(self, in_channels, dropout_rate, Layer_scale_init=1.0e-5):
        super().__init__()
        self.net1 = torch.nn.Sequential(
            torch.nn.LayerNorm(in_channels),
            torch.nn.Linear(in_channels, in_channels*6))
        self.depthwise = torch.nn.Conv1d(in_channels*6, in_channels*6, 3, padding=1, groups=in_channels*6)
        self.net2 = torch.nn.Sequential(
            torch.nn.GLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(in_channels*3, in_channels),
            torch.nn.Dropout(dropout_rate))
        self.Layer_scale = LayerScale(dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init)
        
    def forward(self, x):
        y = self.net1(x)
        y = y.permute(0, 2, 1).contiguous()
        y = self.depthwise(y)
        y = y.permute(0, 2, 1).contiguous()
        y = self.net2(y)
        return x + self.Layer_scale(y)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, in_channels, dropout_rate, layer_scale_init=1.0e-5):
        super().__init__()
        assert in_channels % n_head == 0
        self.d_k = in_channels // n_head
        self.h = n_head
        self.layer_norm = torch.nn.LayerNorm(in_channels)
        self.linear_q = torch.nn.Linear(in_channels, in_channels)
        self.linear_k = torch.nn.Linear(in_channels, in_channels)
        self.linear_v = torch.nn.Linear(in_channels, in_channels)
        self.linear_out = torch.nn.Linear(in_channels, in_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.Layer_scale = LayerScale(dims=3, input_size=in_channels, Layer_scale_init=layer_scale_init)

    def forward(self, q, k, v, mask=None, pos_k=None):
        n_batch = q.size(0)

        q = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)

        q = self.linear_q(q).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(k).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(v).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if pos_k is not None:
            # pos_k的形状是 (seq_len, seq_len, embedding_dim)
            # 扩展到 (batch_size, num_heads, seq_len, seq_len)
            pos_k = pos_k.unsqueeze(0).unsqueeze(0).expand(n_batch, self.h, -1, -1, -1)
            pos_k = pos_k[:, :, :scores.size(-2), :scores.size(-1), 0]  # 选择第一个embedding维度
            scores = scores + pos_k

        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0).expand(n_batch, self.h, -1, -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, v).transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.Layer_scale(self.dropout(self.linear_out(x))), attn
class StreamingTransformer(torch.nn.Module):
    def __init__(self, n_head, in_channels, dropout_rate, num_stage):
        super().__init__()
        self.n_head = n_head
        self.in_channels = in_channels
        self.d_k = in_channels // n_head
        self.attention_layer = MultiHeadAttention(n_head, in_channels, dropout_rate)
        self.history_size = 400
        self.history_k = None
        self.history_v = None
        self.pe_k = torch.nn.Embedding(num_embeddings=2 * self.history_size, embedding_dim=1)

    def forward(self, x):
        n_batch, seq_len, _ = x.size()
        outputs = []

        q = x
        if self.history_k is not None and self.history_v is not None:
            k = torch.cat([self.history_k, x], dim=1)
            v = torch.cat([self.history_v, x], dim=1)
        else:
            k, v = x, x

        pos_seq = torch.arange(0, k.size(1)).long().to(x.device)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]
        pos_seq.clamp_(-self.history_size, self.history_size - 1)
        pos_seq += self.history_size
        pos_k = self.pe_k(pos_seq)
        mask = torch.ones(seq_len, k.size(1)).to(x.device)
        out, attn = self.attention_layer(q, k, v, mask, pos_k)

        if self.history_k is None:
            self.history_k = k
            self.history_v = v
        else:
            self.history_k = k[:, -self.history_size:] if k.size(1) > self.history_size else k
            self.history_v = v[:, -self.history_size:] if v.size(1) > self.history_size else v

        outputs.append(out)

        return outputs[-1]

class EGA(torch.nn.Module):
    def __init__(self, in_channels: int, num_mha_heads: int, dropout_rate: float, num_stage:int):
        super().__init__()
        self.block = torch.nn.ModuleDict({
            'self_attn': StreamingTransformer(
                n_head=num_mha_heads, in_channels=in_channels, dropout_rate=dropout_rate, num_stage= num_stage),
            'linear': torch.nn.Sequential(
                torch.nn.LayerNorm(normalized_shape=in_channels), 
                torch.nn.Linear(in_features=in_channels, out_features=in_channels), 
                torch.nn.Sigmoid())
        })
    
    def forward(self, x: torch.Tensor):
        """
        Compute encoded features.
            :param torch.Tensor x: encoded source features (batch, max_time_in, size)
            :param torch.Tensor mask: mask for x (batch, max_time_in)
            :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        down_len = x.shape[-1]
        x_down = torch.nn.functional.adaptive_avg_pool1d(input=x, output_size=down_len)
        x = x.permute([0, 2, 1])
        x_down = x_down.permute([0, 2, 1])
        x_down = self.block['self_attn'](x_down)
        x_down = x_down.permute([0, 2, 1])
        x_downup = torch.nn.functional.upsample(input=x_down, size=x.shape[1])
        x_downup = x_downup.permute([0, 2, 1])
        x = x + self.block['linear'](x) * x_downup

        return x



class CLA(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, dropout_rate, Layer_scale_init=1.0e-5):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(in_channels)
        self.linear1 = torch.nn.Linear(in_channels, in_channels*2)
        self.GLU = torch.nn.GLU()
        self.dw_conv_1d = torch.nn.Conv1d(in_channels, in_channels, kernel_size, padding='same', groups=in_channels)
        self.linear2 = torch.nn.Linear(in_channels, 2*in_channels)        
        self.BN = torch.nn.BatchNorm1d(2*in_channels)
        self.linear3 = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Linear(2*in_channels, in_channels),
            torch.nn.Dropout(dropout_rate))
        self.Layer_scale = LayerScale(dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init)
    
    def forward(self, x):
        y = self.layer_norm(x)
        y = self.linear1(y)
        y = self.GLU(y)
        y = y.permute([0, 2, 1]) # B, F, T
        y = self.dw_conv_1d(y)
        y = y.permute(0, 2, 1) # B, T, 2F
        y = self.linear2(y)
        y = y.permute(0, 2, 1) # B, T, 2F
        y = self.BN(y)
        y = y.permute(0, 2, 1) # B, T, 2F        
        y = self.linear3(y)
        
        return x + self.Layer_scale(y)
    
class GlobalBlock(torch.nn.Module):
    def __init__(self, in_channels: int, num_mha_heads: int, dropout_rate: float, num_stage:int):
        super().__init__()
        self.block = torch.nn.ModuleDict({
            'ega': EGA(
                num_mha_heads=num_mha_heads, in_channels=in_channels, dropout_rate=dropout_rate, num_stage=num_stage),
            'gcfn': GCFN(in_channels=in_channels, dropout_rate=dropout_rate)
        })
    
    def forward(self, x: torch.Tensor):
        """
        Compute encoded features.
            :param torch.Tensor x: encoded source features (batch, max_time_in, size)
            :param torch.Tensor mask: mask for x (batch, max_time_in)
            :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        x = self.block['ega'](x)
        x = self.block['gcfn'](x)
        x = x.permute([0, 2, 1])

        return x


class LocalBlock(torch.nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.block = torch.nn.ModuleDict({
            'cla': CLA(in_channels, kernel_size, dropout_rate),
            'gcfn': GCFN(in_channels, dropout_rate)
        })
    
    def forward(self, x: torch.Tensor):
        x = self.block['cla'](x)
        x = self.block['gcfn'](x)

        return x
    
    
class SpkAttention(torch.nn.Module):
    def __init__(self, in_channels: int, num_mha_heads: int, dropout_rate: float, num_stage: int):
        super().__init__()
        self.self_attn = StreamingTransformer(n_head=num_mha_heads, in_channels=in_channels, dropout_rate=dropout_rate, num_stage=num_stage)
        self.feed_forward = GCFN(in_channels=in_channels, dropout_rate=dropout_rate)
    
    def forward(self, x: torch.Tensor, num_spk: int):
        """
        Compute encoded features.
            :param torch.Tensor x: encoded source features (batch, max_time_in, size)
            :param torch.Tensor mask: mask for x (batch, max_time_in)
            :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        B, F, T = x.shape
        x = x.view(B//num_spk, num_spk, F, T).contiguous()
        x = x.permute([0, 3, 1, 2]).contiguous()
        x = x.view(-1, num_spk, F).contiguous()
        x = x + self.self_attn(x)
        x = x.view(B//num_spk, T, num_spk, F).contiguous()
        x = x.permute([0, 2, 3, 1]).contiguous()
        x = x.view(B, F, T).contiguous()
        x = x.permute([0, 2, 1])
        x = self.feed_forward(x)
        x = x.permute([0, 2, 1])

        return x
