"""
Config-driven augmentation builders.
"""

from __future__ import annotations

from typing import Any

from torchvision import transforms


def _maybe_add_resize(ops: list, section_cfg: dict[str, Any]) -> None:
    resize = section_cfg.get("resize")
    if resize is not None:
        ops.append(transforms.Resize(tuple(resize)))


def _maybe_add_random_resized_crop(ops: list, section_cfg: dict[str, Any]) -> None:
    cfg = section_cfg.get("random_resized_crop")
    if cfg:
        ops.append(
            transforms.RandomResizedCrop(
                size=cfg["size"],
                scale=tuple(cfg.get("scale", [0.8, 1.0])),
                ratio=tuple(cfg.get("ratio", [0.75, 1.33])),
            )
        )


def _maybe_add_horizontal_flip(ops: list, section_cfg: dict[str, Any]) -> None:
    cfg = section_cfg.get("horizontal_flip")
    if cfg:
        ops.append(transforms.RandomHorizontalFlip(p=float(cfg.get("p", 0.5))))


def _maybe_add_vertical_flip(ops: list, section_cfg: dict[str, Any]) -> None:
    cfg = section_cfg.get("vertical_flip")
    if cfg:
        ops.append(transforms.RandomVerticalFlip(p=float(cfg.get("p", 0.5))))


def _maybe_add_rotation(ops: list, section_cfg: dict[str, Any]) -> None:
    cfg = section_cfg.get("rotation")
    if cfg:
        ops.append(transforms.RandomRotation(degrees=float(cfg.get("degrees", 15))))


def _maybe_add_color_jitter(ops: list, section_cfg: dict[str, Any]) -> None:
    cfg = section_cfg.get("color_jitter")
    if cfg:
        ops.append(
            transforms.ColorJitter(
                brightness=float(cfg.get("brightness", 0.0)),
                contrast=float(cfg.get("contrast", 0.0)),
                saturation=float(cfg.get("saturation", 0.0)),
                hue=float(cfg.get("hue", 0.0)),
            )
        )


def _maybe_add_affine(ops: list, section_cfg: dict[str, Any]) -> None:
    cfg = section_cfg.get("affine")
    if cfg:
        ops.append(
            transforms.RandomAffine(
                degrees=0,
                translate=tuple(cfg.get("translate", [0.0, 0.0])),
                scale=tuple(cfg.get("scale", [1.0, 1.0])),
                shear=float(cfg.get("shear", 0.0)),
            )
        )


def _maybe_add_gaussian_blur(ops: list, section_cfg: dict[str, Any]) -> None:
    cfg = section_cfg.get("gaussian_blur")
    if cfg:
        ops.append(
            transforms.GaussianBlur(
                kernel_size=int(cfg.get("kernel_size", 3)),
                sigma=tuple(cfg.get("sigma", [0.1, 1.0])),
            )
        )


def _maybe_add_to_tensor(ops: list) -> None:
    ops.append(transforms.ToTensor())


def _maybe_add_random_erasing(ops: list, section_cfg: dict[str, Any]) -> None:
    cfg = section_cfg.get("random_erasing")
    if cfg:
        ops.append(
            transforms.RandomErasing(
                p=float(cfg.get("p", 0.5)),
                scale=tuple(cfg.get("scale", [0.02, 0.33])),
                ratio=tuple(cfg.get("ratio", [0.3, 3.3])),
            )
        )


def _maybe_add_normalize(ops: list, section_cfg: dict[str, Any]) -> None:
    cfg = section_cfg.get("normalize")
    if cfg:
        ops.append(transforms.Normalize(mean=tuple(cfg["mean"]), std=tuple(cfg["std"])))


def build_transform_from_section(section_cfg: dict[str, Any]) -> transforms.Compose:
    """Build a torchvision transform pipeline from one config section."""
    ops: list[Any] = []

    _maybe_add_resize(ops, section_cfg)
    _maybe_add_random_resized_crop(ops, section_cfg)
    _maybe_add_horizontal_flip(ops, section_cfg)
    _maybe_add_vertical_flip(ops, section_cfg)
    _maybe_add_rotation(ops, section_cfg)
    _maybe_add_color_jitter(ops, section_cfg)
    _maybe_add_affine(ops, section_cfg)
    _maybe_add_gaussian_blur(ops, section_cfg)
    _maybe_add_to_tensor(ops)
    _maybe_add_random_erasing(ops, section_cfg)
    _maybe_add_normalize(ops, section_cfg)

    return transforms.Compose(ops)


def build_train_val_test_transforms(aug_cfg: dict[str, Any]) -> dict[str, transforms.Compose]:
    """
    Build train/val/test transforms from augmentation config.

    Expected keys:
        augmentation:
            ...
        train:
            ...
        val_test:
            ...
    """
    train_cfg = aug_cfg["train"]
    val_test_cfg = aug_cfg["val_test"]

    return {
        "train": build_transform_from_section(train_cfg),
        "val": build_transform_from_section(val_test_cfg),
        "test": build_transform_from_section(val_test_cfg),
    }