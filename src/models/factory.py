"""
Unified model factory for MaizeFormerX and all baselines.
"""

from __future__ import annotations

from typing import Any

from src.constants import (
    MODEL_EFFICIENTFORMER,
    MODEL_EFFICIENTNET_B0,
    MODEL_EFFICIENTNET_B1,
    MODEL_GHOSTNET,
    MODEL_MAIZEFORMERX,
    MODEL_MIXMOBILENET,
    MODEL_MOBILENETV3,
    MODEL_MOBILEVIT,
    MODEL_SHUFFLENETV2,
    MODEL_SWINV2,
    MODEL_TINYVIT,
)
from src.models.baselines.efficientformer import EfficientFormerBaseline
from src.models.baselines.efficientnet import EfficientNetBaseline
from src.models.baselines.ghostnet import GhostNetBaseline
from src.models.baselines.mixmobilenet import MixMobileNet
from src.models.baselines.mobilenetv3 import MobileNetV3Baseline
from src.models.baselines.mobilevit import MobileViTBaseline
from src.models.baselines.shufflenetv2 import ShuffleNetV2Baseline
from src.models.baselines.swinv2 import SwinV2Baseline
from src.models.baselines.tinyvit import TinyViTBaseline
from src.models.maizeformerx.ablations import (
    MaizeFormerXNoCSAF,
    MaizeFormerXNoMultiScale,
    MaizeFormerXNoTransformer,
)
from src.models.maizeformerx.model import MaizeFormerX


def build_maizeformerx_from_config(model_cfg: dict[str, Any], num_classes: int):
    model = model_cfg["model"]

    patch_embed = model["patch_embed"]
    fusion = model["fusion"]
    encoder = model["encoder"]
    classifier = model["classifier"]
    init = model.get("init", {"std": 0.02})

    return MaizeFormerX(
        num_classes=num_classes,
        in_channels=model.get("in_channels", 3),
        image_size=model.get("image_size", 224),
        dropout=model.get("dropout", 0.3),
        drop_path=model.get("drop_path", 0.1),
        patch_sizes=patch_embed["patch_sizes"],
        strides=patch_embed["strides"],
        embed_dims=patch_embed["embed_dims"],
        fusion_dim=fusion["fusion_dim"],
        fusion_heads=fusion["num_heads"],
        fusion_qkv_bias=fusion.get("qkv_bias", True),
        fusion_attn_dropout=fusion.get("attn_dropout", 0.0),
        fusion_proj_dropout=fusion.get("proj_dropout", 0.0),
        fusion_residual=fusion.get("residual", True),
        encoder_depth=encoder["depth"],
        encoder_num_heads=encoder["num_heads"],
        encoder_mlp_ratio=encoder.get("mlp_ratio", 4.0),
        encoder_qkv_bias=encoder.get("qkv_bias", True),
        encoder_proj_dropout=encoder.get("proj_dropout", 0.0),
        encoder_attn_dropout=encoder.get("attn_dropout", 0.0),
        encoder_layer_scale=encoder.get("layer_scale", False),
        norm_layer=encoder.get("norm_layer", "layernorm"),
        pooling=classifier.get("pooling", "cls"),
        head_dropout=classifier.get("head_dropout", 0.0),
        init_std=init.get("std", 0.02),
    )


def build_model(model_name: str, model_cfg: dict[str, Any], num_classes: int):
    """
    Unified model constructor from config.
    """
    cfg = model_cfg["model"]

    if model_name == MODEL_MAIZEFORMERX:
        return build_maizeformerx_from_config(model_cfg, num_classes)

    if model_name == MODEL_MOBILEVIT:
        return MobileViTBaseline(
            num_classes=num_classes,
            variant=cfg.get("variant", "mobilevit_xs"),
            pretrained=cfg.get("pretrained", False),
            in_chans=cfg.get("in_channels", 3),
            drop_rate=cfg.get("dropout", 0.2),
            drop_path_rate=cfg.get("drop_path", 0.05),
        )

    if model_name == MODEL_EFFICIENTFORMER:
        return EfficientFormerBaseline(
            num_classes=num_classes,
            variant=cfg.get("variant", "efficientformer_l1"),
            pretrained=cfg.get("pretrained", False),
            in_chans=cfg.get("in_channels", 3),
            drop_rate=cfg.get("dropout", 0.2),
            drop_path_rate=cfg.get("drop_path", 0.05),
        )

    if model_name == MODEL_TINYVIT:
        return TinyViTBaseline(
            num_classes=num_classes,
            variant=cfg.get("variant", "tiny_vit_5m_224"),
            pretrained=cfg.get("pretrained", False),
            in_chans=cfg.get("in_channels", 3),
            drop_rate=cfg.get("dropout", 0.2),
            drop_path_rate=cfg.get("drop_path", 0.1),
        )

    if model_name == MODEL_SWINV2:
        return SwinV2Baseline(
            num_classes=num_classes,
            variant=cfg.get("variant", "swinv2_tiny_window8_256"),
            pretrained=cfg.get("pretrained", False),
            in_chans=cfg.get("in_channels", 3),
            drop_rate=cfg.get("dropout", 0.2),
            drop_path_rate=cfg.get("drop_path", 0.1),
        )

    if model_name == MODEL_SHUFFLENETV2:
        return ShuffleNetV2Baseline(
            num_classes=num_classes,
            variant=cfg.get("variant", "shufflenet_v2_x1_0"),
            pretrained=cfg.get("pretrained", False),
            in_chans=cfg.get("in_channels", 3),
        )

    if model_name == MODEL_GHOSTNET:
        return GhostNetBaseline(
            num_classes=num_classes,
            variant=cfg.get("variant", "ghostnet_100"),
            pretrained=cfg.get("pretrained", False),
            in_chans=cfg.get("in_channels", 3),
        )

    if model_name == MODEL_MOBILENETV3:
        return MobileNetV3Baseline(
            num_classes=num_classes,
            variant=cfg.get("variant", "mobilenetv3_large_100"),
            pretrained=cfg.get("pretrained", False),
            in_chans=cfg.get("in_channels", 3),
            drop_rate=cfg.get("dropout", 0.2),
        )

    if model_name in {MODEL_EFFICIENTNET_B0, MODEL_EFFICIENTNET_B1}:
        return EfficientNetBaseline(
            num_classes=num_classes,
            variant=cfg.get("variant", model_name),
            pretrained=cfg.get("pretrained", False),
            in_chans=cfg.get("in_channels", 3),
            drop_rate=cfg.get("dropout", 0.2),
        )

    if model_name == MODEL_MIXMOBILENET:
        backbone_cfg = cfg.get("backbone", {})
        classifier_cfg = cfg.get("classifier", {})
        return MixMobileNet(
            num_classes=num_classes,
            in_chans=cfg.get("in_channels", 3),
            width_mult=backbone_cfg.get("width_mult", 1.0),
            use_se=backbone_cfg.get("use_se", True),
            hidden_dim=classifier_cfg.get("hidden_dim", 512),
            dropout=classifier_cfg.get("dropout", cfg.get("dropout", 0.25)),
        )

    raise ValueError(f"Unsupported model name: {model_name}")


def build_ablation_model(ablation_name: str, model_cfg: dict[str, Any], num_classes: int):
    """
    Construct ablation variants.
    """
    base = model_cfg["model"]
    patch_embed = base.get("patch_embed", {})
    fusion = base.get("fusion", {})
    encoder = base.get("encoder", {})
    classifier = base.get("classifier", {})
    init = base.get("init", {"std": 0.02})

    common_kwargs = dict(
        num_classes=num_classes,
        in_channels=base.get("in_channels", 3),
        image_size=base.get("image_size", 224),
        dropout=base.get("dropout", 0.3),
        drop_path=base.get("drop_path", 0.1),
        patch_sizes=patch_embed.get("patch_sizes", [8, 16, 32]),
        strides=patch_embed.get("strides", [8, 16, 32]),
        embed_dims=patch_embed.get("embed_dims", [64, 96, 128]),
        fusion_dim=fusion.get("fusion_dim", 192),
        fusion_heads=fusion.get("num_heads", 4),
        fusion_qkv_bias=fusion.get("qkv_bias", True),
        fusion_attn_dropout=fusion.get("attn_dropout", 0.1),
        fusion_proj_dropout=fusion.get("proj_dropout", 0.1),
        fusion_residual=fusion.get("residual", True),
        encoder_depth=encoder.get("depth", 6),
        encoder_num_heads=encoder.get("num_heads", 6),
        encoder_mlp_ratio=encoder.get("mlp_ratio", 4.0),
        encoder_qkv_bias=encoder.get("qkv_bias", True),
        encoder_proj_dropout=encoder.get("proj_dropout", 0.1),
        encoder_attn_dropout=encoder.get("attn_dropout", 0.1),
        encoder_layer_scale=encoder.get("layer_scale", False),
        norm_layer=encoder.get("norm_layer", "layernorm"),
        pooling=classifier.get("pooling", "cls"),
        head_dropout=classifier.get("head_dropout", 0.3),
        init_std=init.get("std", 0.02),
    )

    if ablation_name == "full_model":
        return MaizeFormerX(**common_kwargs)
    if ablation_name == "without_csaf":
        return MaizeFormerXNoCSAF(**common_kwargs)
    if ablation_name == "without_multi_scale":
        return MaizeFormerXNoMultiScale(
            num_classes=num_classes,
            in_channels=base.get("in_channels", 3),
            image_size=base.get("image_size", 224),
            patch_size=16,
            embed_dim=fusion.get("fusion_dim", 192),
            encoder_depth=encoder.get("depth", 6),
            encoder_num_heads=encoder.get("num_heads", 6),
            encoder_mlp_ratio=encoder.get("mlp_ratio", 4.0),
            dropout=base.get("dropout", 0.3),
            drop_path=base.get("drop_path", 0.1),
            pooling=classifier.get("pooling", "cls"),
            head_dropout=classifier.get("head_dropout", 0.3),
        )
    if ablation_name == "without_transformer_encoder":
        return MaizeFormerXNoTransformer(**common_kwargs)

    raise ValueError(f"Unsupported ablation name: {ablation_name}")