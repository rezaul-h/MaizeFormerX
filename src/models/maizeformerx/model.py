"""
Main MaizeFormerX model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common.layers import trunc_normal_
from src.models.maizeformerx.classifier import MaizeFormerXClassifier
from src.models.maizeformerx.csaf import CSAF
from src.models.maizeformerx.encoder import TransformerEncoder
from src.models.maizeformerx.multi_scale_patch_embed import MultiScalePatchEmbed


class MaizeFormerX(nn.Module):
    """
    MaizeFormerX:
      multi-scale patch embedding -> CSAF -> cls token + encoder -> classifier
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        image_size: int = 224,
        dropout: float = 0.3,
        drop_path: float = 0.1,
        patch_sizes: list[int] | None = None,
        strides: list[int] | None = None,
        embed_dims: list[int] | None = None,
        fusion_dim: int = 192,
        fusion_heads: int = 4,
        fusion_qkv_bias: bool = True,
        fusion_attn_dropout: float = 0.1,
        fusion_proj_dropout: float = 0.1,
        fusion_residual: bool = True,
        encoder_depth: int = 6,
        encoder_num_heads: int = 6,
        encoder_mlp_ratio: float = 4.0,
        encoder_qkv_bias: bool = True,
        encoder_proj_dropout: float = 0.1,
        encoder_attn_dropout: float = 0.1,
        encoder_layer_scale: bool = False,
        norm_layer: str = "layernorm",
        pooling: str = "cls",
        head_dropout: float = 0.3,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()

        patch_sizes = patch_sizes or [8, 16, 32]
        strides = strides or [8, 16, 32]
        embed_dims = embed_dims or [64, 96, 128]

        self.image_size = image_size
        self.num_classes = num_classes
        self.embed_dim = fusion_dim
        self.pooling = pooling

        self.patch_embed = MultiScalePatchEmbed(
            in_channels=in_channels,
            patch_sizes=patch_sizes,
            strides=strides,
            embed_dims=embed_dims,
            norm_layer=norm_layer,
            flatten=True,
            use_conv_projection=True,
        )

        self.fusion = CSAF(
            input_dims=embed_dims,
            fusion_dim=fusion_dim,
            num_heads=fusion_heads,
            qkv_bias=fusion_qkv_bias,
            attn_dropout=fusion_attn_dropout,
            proj_dropout=fusion_proj_dropout,
            residual=fusion_residual,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.encoder = TransformerEncoder(
            depth=encoder_depth,
            dim=fusion_dim,
            num_heads=encoder_num_heads,
            mlp_ratio=encoder_mlp_ratio,
            qkv_bias=encoder_qkv_bias,
            proj_dropout=encoder_proj_dropout,
            attn_dropout=encoder_attn_dropout,
            drop_path=drop_path,
            layer_scale=encoder_layer_scale,
            norm_layer=norm_layer,
        )

        self.classifier = MaizeFormerXClassifier(
            embed_dim=fusion_dim,
            num_classes=num_classes,
            pooling=pooling,
            head_dropout=head_dropout,
        )

        self._init_weights(init_std)

    def _init_weights(self, std: float) -> None:
        trunc_normal_(self.cls_token, std=std)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale_tokens = self.patch_embed(x)
        fused_tokens = self.fusion(multi_scale_tokens)

        b = fused_tokens.shape[0]
        cls_token = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls_token, fused_tokens], dim=1)
        tokens = self.pos_drop(tokens)

        encoded = self.encoder(tokens)
        return encoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits