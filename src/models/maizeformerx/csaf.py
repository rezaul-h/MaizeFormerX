"""
Cross-Scale Attention Fusion (CSAF) for MaizeFormerX.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common.attention import MultiHeadSelfAttention
from src.models.common.layers import init_linear
from src.models.common.norms import get_norm_layer


class CSAF(nn.Module):
    """
    Cross-Scale Attention Fusion.

    Each scale is first projected to a shared fusion dimension. Tokens from all
    scales are concatenated, processed by self-attention, and then summarized
    into a fused token sequence.

    Current output:
        fused tokens of shape [B, N_total, fusion_dim]
    """

    def __init__(
        self,
        input_dims: list[int],
        fusion_dim: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.fusion_dim = fusion_dim
        self.residual = residual

        self.projections = nn.ModuleList(
            [nn.Linear(in_dim, fusion_dim) for in_dim in input_dims]
        )
        self.norm = get_norm_layer("layernorm", fusion_dim)
        self.attn = MultiHeadSelfAttention(
            dim=fusion_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            init_linear(m)

    def forward(self, multi_scale_tokens: list[torch.Tensor]) -> torch.Tensor:
        if len(multi_scale_tokens) != len(self.projections):
            raise ValueError("Number of input token sets must match number of input_dims.")

        projected = [
            proj(tokens) for proj, tokens in zip(self.projections, multi_scale_tokens)
        ]
        fused = torch.cat(projected, dim=1)  # B, sum(N_i), fusion_dim

        out = self.attn(self.norm(fused))
        if self.residual:
            out = out + fused
        return out