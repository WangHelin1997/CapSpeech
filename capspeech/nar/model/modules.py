"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations
from typing import Optional
import math
from torch.utils.checkpoint import checkpoint

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

from einops import rearrange
# from x_transformers.x_transformers import apply_rotary_pos_emb
from inspect import isfunction
from torch.amp import autocast


# raw wav to mel spec

class MelSpec(torch.nn.Module):
    def __init__(self, target_sample_rate=24000, filter_length=1024, hop_length=256, n_mel_channels=100, f_min=0, f_max=12000, normalize=False, power=1, norm=None, center=True,):
        super().__init__()
        self.frame_length = filter_length
        self.hop_length = hop_length
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=filter_length,
            win_length=filter_length,
            hop_length=hop_length,
            center=False,
            power=1.0,
            norm="slaney",
            n_mels=n_mel_channels,
            mel_scale="slaney",
            f_min=0,
            f_max=12000
        )

    @torch.no_grad()
    def forward(self, x, target_length=None):
        if len(x.shape) == 3:
            x = rearrange(x, 'b 1 nw -> b nw')
        assert len(x.shape) == 2
        x = F.pad(x, ((self.frame_length - self.hop_length) // 2,
                      (self.frame_length - self.hop_length) // 2), "reflect")
        mel = self.mel(x)

        target_length = mel.shape[-1] if target_length is None else target_length
        logmel = torch.zeros(mel.shape[0], mel.shape[1], target_length).to(mel.device)
        logmel[:, :, :mel.shape[2]] = mel

        logmel = torch.log(torch.clamp(logmel, min=1e-5))
        return logmel

# class MelSpec(nn.Module):
#     def __init__(
#             self,
#             filter_length=1024,
#             hop_length=256,
#             win_length=1024,
#             n_mel_channels=100,
#             target_sample_rate=24_000,
#             normalize=False,
#             power=2,
#             norm='slaney',
#             center=True,
#             mel_scale='slaney',
#     ):
#         super().__init__()
#         self.n_mel_channels = n_mel_channels

#         self.mel_stft = torchaudio.transforms.MelSpectrogram(
#             sample_rate=target_sample_rate,
#             n_fft=filter_length,
#             win_length=win_length,
#             hop_length=hop_length,
#             n_mels=n_mel_channels,
#             power=power,
#             center=center,
#             normalized=normalize,
#             norm=norm,
#             mel_scale=mel_scale
#         )

#         self.register_buffer('dummy', torch.tensor(0), persistent=False)

#     def forward(self, inp):
#         if len(inp.shape) == 3:
#             inp = rearrange(inp, 'b 1 nw -> b nw')

#         assert len(inp.shape) == 2

#         if self.dummy.device != inp.device:
#             self.to(inp.device)

#         mel = self.mel_stft(inp)
#         mel = mel.clamp(min=1e-5).log()
#         return mel


# sinusoidal position embedding

class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# convolutional position embedding

class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: float['b n d'], mask: bool['b n'] | None = None):
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.)

        x = rearrange(x, 'b n d -> b d n')
        x = self.conv1d(x)
        out = rearrange(x, 'b d n -> b n d')

        if mask is not None:
            out = out.masked_fill(~mask, 0.)

        return out


# rotary positional embedding related

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = start.unsqueeze(1) + (
            torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) *
            scale.unsqueeze(1)).long()
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


# Global Response Normalization layer (Instance Normalization ?)

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-V2 Block https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
# ref: https://github.com/bfs18/e2_tts/blob/main/rfwave/modules.py#L108

class ConvNeXtV2Block(nn.Module):
    def __init__(
            self,
            dim: int,
            intermediate_dim: int,
            dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding,
                                groups=dim, dilation=dilation)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# AdaLayerNormZero
# return with modulated x for attn input, and params for later mlp modulation

class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNormZero for final layer
# return only with modulated x for attn input, cuz no more mlp modulation

class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# FeedForward

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.,
                 approximate: str = 'none'):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            activation
        )
        self.ff = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.ff(x)


# Attention with possible joint part
# modified from diffusers/src/diffusers/models/attention_processor.py

class Attention(nn.Module):
    def __init__(
            self,
            processor: AttnProcessor,
            dim: int,
            heads: int = 8,
            dim_head: int = 64,
            dropout: float = 0.0,
            qk_norm: bool = True,
            # context_dim: Optional[int] = None,  # if not None -> joint attention
            # context_pre_only=None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention equires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.processor = processor

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        # self.context_dim = context_dim
        # self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm is True:
            self.q_norm = nn.LayerNorm(dim_head, eps=1e-6)
            self.k_norm = nn.LayerNorm(dim_head, eps=1e-6)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")

        # if self.context_dim is not None:
        #     self.to_k_c = nn.Linear(context_dim, self.inner_dim)
        #     self.to_v_c = nn.Linear(context_dim, self.inner_dim)
        #     if self.context_pre_only is not None:
        #         self.to_q_c = nn.Linear(context_dim, self.inner_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        # if self.context_pre_only is not None and not self.context_pre_only:
        #     self.to_out_c = nn.Linear(self.inner_dim, dim)

    def forward(self, x, c=None, mask=None,
                rope=None, c_rope=None, ) -> torch.Tensor:
        # if c is not None:
        #     return self.processor(self, x, c = c, mask = mask, rope = rope, c_rope = c_rope)
        # else:
        #     return self.processor(self, x, mask = mask, rope = rope)
        return self.processor(self, x=x, c=c,
                              mask=mask, rope=rope, c_rope=c_rope)


# Attention processor

def create_mask(q_shape, k_shape, device, q_mask=None, k_mask=None):
    def default(val, d):
        return val if val is not None else (d() if isfunction(d) else d)

    b, i, j, device = q_shape[0], q_shape[-2], k_shape[-2], device
    q_mask = default(q_mask, torch.ones((b, i), device=device, dtype=torch.bool))
    k_mask = default(k_mask, torch.ones((b, j), device=device, dtype=torch.bool))
    attn_mask = rearrange(q_mask, 'b i -> b 1 i 1') * rearrange(k_mask, 'b j -> b 1 1 j')
    return attn_mask


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


@autocast('cuda', enabled = False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    freqs = freqs[:, -seq_len:, :]
    scale = scale[:, -seq_len:, :] if isinstance(scale, torch.Tensor) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t, t_unrotated), dim = -1)

    return out.type(orig_dtype)


class AttnProcessor:
    def __init__(self):
        pass

    def __call__(
            self,
            attn: Attention,
            x: float['b n d'],  # noised input x
            mask: bool['b n'] | None = None,
            rope=None,  # rotary position embedding
            c=None,  # context
            c_rope=None,  # context rope
    ) -> torch.FloatTensor:

        batch_size = x.shape[0]

        if c is None:
            c = x

        # `sample` projections.
        query = attn.to_q(x)
        key = attn.to_k(c)
        value = attn.to_v(c)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale ** -1.0) if xpos_scale is not None else (1.0, 1.0)

            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        # if mask is not None:
        #     attn_mask = mask
        #     attn_mask = rearrange(attn_mask, 'b n -> b 1 1 n')
        #     attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        # else:
        #     attn_mask = None
        if mask is not None:
            attn_mask = create_mask(x.shape, c.shape,
                                    x.device, None, mask)
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask,
                                           dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)

        # if mask is not None:
        #     mask = rearrange(mask, 'b n -> b n 1')
        #     x = x.masked_fill(~mask, 0.)

        return x


# DiT Block

class DiTBlock(nn.Module):

    def __init__(self, dim, heads, dim_head,
                 ff_mult=4, dropout=0.1,
                 qk_norm=False,
                 use_checkpoint=True):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            processor=AttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult,
                              dropout=dropout, approximate="tanh")

        self.use_checkpoint = checkpoint

    def forward(self, x, t, mask=None, rope=None):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, t, mask, rope)
        else:
            return self._forward(x, t, mask, rope)

    # x: noised input, t: time embedding
    def _forward(self, x, t, mask=None, rope=None):
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


# Cross DiT Block
class CrossDiTBlock(nn.Module):

    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1,
                 qk_norm=False,
                 use_checkpoint=True, skip=False):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            processor=AttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
        )

        self.cross_norm = nn.LayerNorm(dim, eps=1e-6)
        self.context_norm = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attn = Attention(
            processor=AttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
        )

        # Zero out the weight
        nn.init.constant_(self.cross_attn.to_out[0].weight, 0.0)
        # Zero out the bias if it exists
        if self.cross_attn.to_out[0].bias is not None:
            nn.init.constant_(self.cross_attn.to_out[0].bias, 0.0)

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

        self.use_checkpoint = checkpoint

        self.skip = skip
        if self.skip:
            self.skip_norm = nn.LayerNorm(dim*2, eps=1e-6)
            self.skip_linear = nn.Linear(dim*2, dim)

    def forward(self, x, t, mask=None, rope=None,
                context=None, context_mask=None, skip=None):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, t, mask, rope, context, context_mask, skip, use_reentrant=False)
        else:
            return self._forward(x, t, mask, rope, context, context_mask, skip)

    def _forward(self, x, t, mask=None, rope=None,
                 context=None, context_mask=None, skip=None):
        if self.skip:
            cat = torch.cat([x, skip], dim=-1)
            cat = self.skip_norm(cat)
            x = self.skip_linear(cat)

        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        # process cross attention
        x = x + self.cross_attn(x=self.cross_norm(x), c=self.context_norm(context),
                                mask=context_mask, rope=None)

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


# time step conditioning embedding

class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, timestep: float['b']):
        time_hidden = self.time_embed(timestep)
        time = self.time_mlp(time_hidden)  # b d
        return time
