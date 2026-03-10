"""
Transformer encoder blocks for MaizeFormerX.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common.attention import MultiHeadSelfAttention
from src.models.common.layers import DropPath, LayerScale
from src.models.common.mlp import MLP
from src.models.common.norms import get_norm_layer


class TransformerEncoderBlock(nn.Module):
    """
    Pre-norm transformer encoder block.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: bool = False,
        norm_layer: str = "layernorm",
    ) -> None:
        super().__init__()
        self.norm1 = get_norm_layer(norm_layer, dim)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.drop_path1 = DropPath(drop_path)
        self.ls1 = LayerScale(dim) if layer_scale else nn.Identity()

        self.norm2 = get_norm_layer(norm_layer, dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_dim,
            out_features=dim,
            dropout=proj_dropout,
        )
        self.drop_path2 = DropPath(drop_path)
        self.ls2 = LayerScale(dim) if layer_scale else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of transformer encoder blocks.
    """

    def __init__(
        self,
        depth: int,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: bool = False,
        norm_layer: str = "layernorm",
    ) -> None:
        super().__init__()

        dpr = torch.linspace(0, drop_path, depth).tolist() if depth > 1 else [drop_path]
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_dropout=proj_dropout,
                    attn_dropout=attn_dropout,
                    drop_path=dpr[i],
                    layer_scale=layer_scale,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = get_norm_layer(norm_layer, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x