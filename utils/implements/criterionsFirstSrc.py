import torch
import numpy as np

from math import ceil
from itertools import permutations
from torchaudio.transforms import MelScale
from dataclasses import dataclass, field, fields
from typing import List, Type, Any, Callable, Optional, Union
from loguru import logger
from utils.decorators import *
from mir_eval.separation import bss_eval_sources


# Utility functions
def l2norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim)

def l1norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim, p=1)

@dataclass(slots=True)
class STFTBase(torch.nn.Module):
    """
    Base layer for (i)STFT
    NOTE:
        1) Recommend sqrt_hann window with 2**N frame length, because it 
           could achieve perfect reconstruction after overlap-add
        2) Now haven't consider padding problems yet
    """
    device: torch.device
    frame_length: int
    frame_shift: int
    window: str
    K: torch.nn.Parameter = field(init=False)
    num_bins: int = field(init=False)

    def __post_init__(self):
        super(STFTBase, self).__init__()  # Initialize the torch.nn.Module base class
        K = self._init_kernel(self.frame_length, self.frame_shift)
        self.K = torch.nn.Parameter(K, requires_grad=False).to(self.device)
        self.num_bins = self.K.shape[0] // 2
    
    def _init_kernel(self, frame_len, frame_hop):
        # FFT points
        N = frame_len
        # window
        if self.window == 'hann':
            W = torch.hann_window(frame_len)
        if N//4 == frame_hop:
            const = (2/3)**0.5       
            W = const*W
        elif N//2 == frame_hop:
            W = W**0.5
        S = 0.5 * (N * N / frame_hop)**0.5
        
        # Updated FFT calculation for efficiency
        K = torch.fft.rfft(torch.eye(N) / S, dim=1)[:frame_len]
        K = torch.stack((torch.real(K), torch.imag(K)), dim=2)
        K = torch.transpose(K, 0, 2) * W # 2 x N/2+1 x F
        K = torch.reshape(K, (N + 2, 1, frame_len)) # N+2 x 1 x F
        return K

    def extra_repr(self):
        return (f"window={self.window}, stride={self.frame_shift}, " +
                f"kernel_size={self.K.shape[0]}x{self.K.shape[2]}")

@logger_wraps()
@dataclass(slots=True)
class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """

    def forward(self, x, cplx=False):
        """
        Accept (single or multiple channel) raw waveform and output magnitude and phase
        args
            x: input signal, N x C x S or N x S
        return
            m: magnitude, N x C x F x T or N x F x T
            p: phase, N x C x F x T or N x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} expect 2D/3D tensor, but got {:d}D signal".format(
                    self.__name__, x.dim()))
        # if N x S, reshape N x 1 x S
        N_frame = ceil(x.shape[-1] / self.frame_shift)
        len_padded = N_frame * self.frame_shift
        if x.dim() == 2:
            
            x = torch.cat((x, torch.zeros(x.shape[0], len_padded-x.shape[-1], device=x.device)), dim=-1)
            x = torch.unsqueeze(x, 1)
            # N x 2F x T
            c = torch.nn.functional.conv1d(x, self.K, stride=self.frame_shift, padding=0)
            # N x F x T
            r, i = torch.chunk(c, 2, dim=1)
        else:        
            x = torch.cat((x, torch.zeros(x.shape[0], x.shape[1], len_padded-x.shape[-1])), dim=-1)
            N, C, S = x.shape
            x = x.reshape(N * C, 1, S)
            # NC x 2F x T
            c = torch.nn.functional.conv1d(x, self.K, stride=self.frame_shift, padding=0)
            # N x C x 2F x T
            c = c.reshape(N, C, -1, c.shape[-1])
            # N x C x F x T
            r, i = torch.chunk(c, 2, dim=2)

        if cplx:
            return r, i
        m = (r**2 + i**2 + 1.0e-10)**0.5
        p = torch.atan2(i, r)
        return m, p

@logger_wraps()
@dataclass(slots=True)
class PIT_SISNR_mag:
    device: torch.device
    frame_length: int
    frame_shift: int
    window: str
    num_stages: int
    num_spks: int
    scale_inv: bool
    mel_opt: bool
    
    
    stft: List[Any] = field(init=False)
    mel_fb: Callable[[torch.Tensor], torch.Tensor] = field(init=False)

    def __post_init__(self):
        self.stft = [STFT(self.device, self.frame_length, self.frame_shift, self.window) for _ in range(self.num_stages)]
        self.mel_fb = MelScale(n_mels=80, sample_rate=16000, n_stft=int(self.frame_length / 2) + 1).to(self.device) if self.mel_opt else lambda x: x

    def __repr__(self):
        # __init__
        class_name = self.__class__.__name__
        init_fields = [f for f in fields(self) if f.init]
        field_strs = [f"{field.name}={getattr(self, field.name)!r}" for field in init_fields]

        # __post_init__
        stft_repr = f"stft = [STFT instance for {len(self.stft)} layers]"
        mel_fb_repr = "mel_fb = MelScale" if self.mel_opt else "mel_fb=Identity"
        post_init_reprs = [stft_repr, mel_fb_repr]

        return f"<{class_name}({', '.join(field_strs + post_init_reprs)})>"

    def __call__(self, **kwargs):
        estims = kwargs['estims'][0]  # Only one estimation
        idx = kwargs['idx']
        input_sizes = kwargs["input_sizes"].to(self.device)
        targets = [t.to(self.device) for t in kwargs["target_attr"]]

        def _STFT_Mag_SDR_loss(mix, src, eps=1.0e-12):
            mix_zm = mix - torch.mean(mix, dim=-1, keepdim=True)
            src_zm = src - torch.mean(src, dim=-1, keepdim=True)
            if self.scale_inv:
                scale = torch.sum(mix_zm * src_zm, dim=-1, keepdim=True) / (l2norm(src_zm, keepdim=True) ** 2 + eps)
                src_zm = torch.clamp(scale, min=1e-2) * src_zm
            mix_zm = self.stft[idx](mix_zm.to(self.device))[0]
            src_zm = self.stft[idx](src_zm.to(self.device))[0]
            if self.mel_opt:
                mix_zm = self.mel_fb(mix_zm)
                src_zm = self.mel_fb(src_zm)
            utt_loss = -20 * torch.log10(eps + l2norm(l2norm((src_zm))) / (l2norm(l2norm(mix_zm - src_zm)) + eps))
            return utt_loss

        # Only compute loss for the first target
        first_target = targets[0]  # First target

        loss = _STFT_Mag_SDR_loss(estims, first_target)

        num_utts = input_sizes.shape[0]
        return loss.sum() / num_utts

@logger_wraps()
@dataclass(slots=True)
class PIT_SISNR_time:
    device: torch.device
    num_spks: int
    scale_inv: bool

    def __repr__(self):
        class_name = self.__class__.__name__
        init_fields = [f for f in fields(self) if f.init]
        field_strs = [f"{field.name}={getattr(self, field.name)!r}" for field in init_fields]
        return f"<{class_name}({', '.join(field_strs)})>"

    def __call__(self, **kwargs):
        estims = kwargs['estims'][0]  # Only one estimation
        input_sizes = kwargs["input_sizes"].to(self.device)
        targets = [target.to(self.device) for target in kwargs["target_attr"]]

        def _SDR_loss(mix, src, eps=1.0e-8):
            mix_zm = mix - torch.mean(mix, dim=-1, keepdim=True)
            src_zm = src - torch.mean(src, dim=-1, keepdim=True)
            if self.scale_inv:
                scale_factor = torch.sum(mix_zm * src_zm, dim=-1, keepdim=True) / (
                            l2norm(src_zm, keepdim=True) ** 2 + eps)
                src_zm = torch.clamp(scale_factor, min=1e-2) * src_zm  # Added clamping to avoid very small scale_factor

            utt_loss = -20 * torch.log10(eps + l2norm(src_zm) / (l2norm(mix_zm - src_zm) + eps))
            utt_loss = torch.clamp(utt_loss, min=-30)  # Clamping loss to a minimum value

            return utt_loss

        # Compute loss for the first target
        first_target = targets[0]  # First target
        loss = _SDR_loss(estims, first_target)

        num_utts = input_sizes.shape[0]
        return torch.sum(loss) / num_utts

@logger_wraps()
@dataclass(slots=True)
class PIT_SISNRi:
    device: torch.device
    num_spks: int
    scale_inv: bool

    def __repr__(self):
        class_name = self.__class__.__name__
        init_fields = [f for f in fields(self) if f.init]
        field_strs = [f"{field.name}={getattr(self, field.name)!r}" for field in init_fields]
        return f"<{class_name}({', '.join(field_strs)})>"

    def __call__(self, **kwargs):
        estims = kwargs['estims'][0]  # Only one estimation
        input_sizes = kwargs["input_sizes"].to(self.device)
        targets = [t.to(self.device) for t in kwargs["target_attr"]]
        input = kwargs['mixture'].to(self.device)
        input_zm = input - torch.mean(input, dim=-1, keepdim=True)
        eps = kwargs['eps']

        def _SDR_loss(mix, est, src, input_zm, eps=1.0e-8):
            est_zm = est - torch.mean(est, dim=-1, keepdim=True)
            src_zm = src - torch.mean(src, dim=-1, keepdim=True)

            if self.scale_inv:
                scale_factor = torch.sum(est_zm * src_zm, dim=-1, keepdim=True) / (
                            l2norm(src_zm, keepdim=True) ** 2 + eps)
                src_zm = torch.clamp(scale_factor, min=1e-2) * src_zm

            utt_loss_est = 20 * torch.log10(eps + l2norm(src_zm) / (l2norm(est_zm - src_zm) + eps))

            if self.scale_inv:
                scale_factor_x = torch.sum(input_zm * src_zm, dim=-1, keepdim=True) / (
                            l2norm(src_zm, keepdim=True) ** 2 + eps)
                src_zm_x = torch.clamp(scale_factor_x, min=1e-2) * src_zm
            else:
                src_zm_x = src_zm

            utt_loss_in = 20 * torch.log10(eps + l2norm(src_zm_x) / (l2norm(input_zm - src_zm_x) + eps))

            return utt_loss_est - utt_loss_in

        # Only compute loss for the first target
        first_target = targets[0]  # First target
        loss = _SDR_loss(input_zm, estims, first_target, input_zm, eps)

        num_utts = input_sizes.shape[0]
        return torch.sum(loss) / num_utts, loss

@logger_wraps()
@dataclass(slots=True)
class PIT_SDRi:
    device: torch.device
    dump: int

    def __repr__(self):
        class_name = self.__class__.__name__
        init_fields = [f for f in fields(self) if f.init]
        field_strs = [f"{field.name}={getattr(self, field.name)!r}" for field in init_fields]
        return f"<{class_name}({', '.join(field_strs)})>"

    def __call__(self, **kwargs):
        estims = kwargs['estims'][0].to(self.device)  # Only one estimation
        input_sizes = kwargs["input_sizes"].to(self.device)
        targets = [t.to(self.device) for t in kwargs["target_attr"]]
        first_target = targets[0]  # Only the first target
        input = kwargs['mixture'].to(self.device)

        # Convert to numpy
        targets_np = first_target.cpu().data.numpy()
        estims_np = estims.cpu().data.numpy()
        input_np = input.cpu().data.numpy()

        # Compute SDR
        min_perutt_out, _, _, _ = bss_eval_sources(targets_np, estims_np)
        min_perutt_in, _, _, _ = bss_eval_sources(targets_np, input_np)

        num_utts = input_sizes.shape[0]
        return np.sum(min_perutt_out - min_perutt_in) / num_utts, min_perutt_out - min_perutt_in