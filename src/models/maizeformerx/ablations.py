"""
Ablation variants for MaizeFormerX.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common.heads import LinearHead
from src.models.common.patch_embed import PatchEmbed
from src.models.maizeformerx.model import MaizeFormerX


class MaizeFormerXNoCSAF(MaizeFormerX):
    """
    Replace CSAF with simple concatenation + projection.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        input_dims = self.fusion.input_dims
        fusion_dim = self.fusion.fusion_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim) if len(set(input_dims)) == 1 else nn.Identity()
        )
        self.scale_projs = nn.ModuleList(
            [nn.Linear(dim, fusion_dim) for dim in input_dims]
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale_tokens = self.patch_embed(x)
        projected = [proj(t) for proj, t in zip(self.scale_projs, multi_scale_tokens)]
        fused_tokens = torch.cat(projected, dim=1)

        b = fused_tokens.shape[0]
        cls_token = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls_token, fused_tokens], dim=1)
        tokens = self.pos_drop(tokens)
        return self.encoder(tokens)


class MaizeFormerXNoMultiScale(nn.Module):
    """
    Single-scale variant without multi-scale patch fusion.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        encoder_depth: int = 6,
        encoder_num_heads: int = 6,
        encoder_mlp_ratio: float = 4.0,
        dropout: float = 0.3,
        drop_path: float = 0.1,
        pooling: str = "cls",
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        from src.models.maizeformerx.encoder import TransformerEncoder

        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            stride=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm_layer="layernorm",
            flatten=True,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.encoder = TransformerEncoder(
            depth=encoder_depth,
            dim=embed_dim,
            num_heads=encoder_num_heads,
            mlp_ratio=encoder_mlp_ratio,
            qkv_bias=True,
            proj_dropout=dropout,
            attn_dropout=dropout,
            drop_path=drop_path,
            layer_scale=False,
            norm_layer="layernorm",
        )
        self.pooling = pooling
        self.head = LinearHead(embed_dim, num_classes, dropout=head_dropout)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        b = tokens.shape[0]
        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_token, tokens], dim=1)
        x = self.pos_drop(x)
        x = self.encoder(x)
        pooled = x[:, 0] if self.pooling == "cls" else x.mean(dim=1)
        return self.head(pooled)


class MaizeFormerXNoTransformer(MaizeFormerX):
    """
    Remove the transformer encoder and classify directly from pooled fused tokens.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale_tokens = self.patch_embed(x)
        fused_tokens = self.fusion(multi_scale_tokens)
        b = fused_tokens.shape[0]
        cls_token = self.cls_token.expand(b, -1, -1)
        return torch.cat([cls_token, fused_tokens], dim=1)