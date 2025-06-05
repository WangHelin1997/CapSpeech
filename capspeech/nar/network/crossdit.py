"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from einops import repeat

from x_transformers.x_transformers import RotaryEmbedding

from capspeech.nar.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    CrossDiTBlock,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis, get_pos_embed_indices,
)


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers = 0, conv_mult = 2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)])
        else:
            self.extra_modeling = False

    def forward(self, text: int['b nt'], seq_len, drop_text=False):
        batch, text_len = text.shape[0], text.shape[1]
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        text = F.pad(text, (0, seq_len - text_len), value = 0)

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text) # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding
class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim = out_dim)

    def forward(self, x: float['b n d'], cond: float['b n d'], 
                text_embed: float['b n d'], drop_audio_cond = False):
        if drop_audio_cond or cond is None:  # cfg for cond audio
            cond = torch.zeros_like(x)

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks

class CrossDiT(nn.Module):
    def __init__(self,
                 dim, depth=8, heads=8, dim_head=64, dropout=0.0, ff_mult=4,
                 mel_dim=100, t5_dim=512, clap_dim=512,
                 text_num_embeds=256, text_dim=None, conv_layers=0,
                 skip=False, use_checkpoint=True, qk_norm=True,
                 ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.caption_embedding = nn.Sequential(
                nn.Linear(t5_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
            )

        self.clap_embedding = nn.Sequential(
                nn.Linear(clap_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, text_dim)
            )

        # self.null_clap = nn.Parameters
        # self.null_prompt = nn.Parameters

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.skip = skip

        self.in_blocks = nn.ModuleList([
            CrossDiTBlock(dim=dim,
                          heads=heads,
                          dim_head=dim_head,
                          ff_mult=ff_mult,
                          dropout=dropout,
                          use_checkpoint=use_checkpoint,
                          qk_norm=qk_norm,
                          skip=False
                         )
            for _ in range(depth//2)
        ])

        self.mid_block = CrossDiTBlock(dim=dim,
                                       heads=heads,
                                       dim_head=dim_head,
                                       ff_mult=ff_mult,
                                       dropout=dropout,
                                       use_checkpoint=use_checkpoint,
                                       qk_norm=qk_norm,
                                       skip=False)
        self.out_blocks = nn.ModuleList([
            CrossDiTBlock(dim=dim,
                          heads=heads,
                          dim_head=dim_head,
                          ff_mult=ff_mult,
                          dropout=dropout,
                          use_checkpoint=use_checkpoint,
                          qk_norm=qk_norm,
                          skip=skip
                         )
            for _ in range(depth//2)
        ])

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(
            self,
            x: float['b n d'],  # nosied input audio
            cond: float['b n d'],  # masked cond audio
            prompt: float['b n d'], # speech caption
            clap: float['b n d'], # sound effects
            text: int['b nt'],  # text
            time: float['b'] | float[''],  # time step
            mask: bool['b n'] | None = None,
            prompt_mask: bool['b n'] | None = None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = repeat(time, ' -> b', b=batch)

        t = self.time_embed(time)
        text_embed = self.text_embed(text, seq_len-1)

        prompt_embed = self.caption_embedding(prompt)
        clap_embed = self.clap_embedding(clap).unsqueeze(1)
        text_embed = torch.cat([clap_embed, text_embed], dim=1)

        x = self.input_embed(x, cond, text_embed)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        skips = []
        for i, block in enumerate(self.in_blocks):
            x = block(x, t, mask=mask, rope=rope, 
                      context=prompt_embed, context_mask=prompt_mask)
            if self.skip:
                skips.append(x)
        x = self.mid_block(x, t, mask=mask, rope=rope, 
                           context=prompt_embed, context_mask=prompt_mask)

        for i, block in enumerate(self.out_blocks):
            if self.skip:
                skip = skips.pop()
            else:
                skip = None
            x = block(x, t, mask=mask, rope=rope, 
                      context=prompt_embed, context_mask=prompt_mask, skip=skip)

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
