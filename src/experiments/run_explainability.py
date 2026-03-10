"""
Explainability experiment using Grad-CAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.constants import MODEL_CONFIG_PATHS, OUTPUT_SALIENCY_DIR, TRAIN_CONFIG_DIR
from src.data.augmentations import build_train_val_test_transforms
from src.data.class_maps import build_class_map
from src.data.dataloaders import build_manifest_dataloader
from src.explainability.casebook import build_case_entry, save_casebook
from src.explainability.gradcam import GradCAM
from src.explainability.overlays import cam_to_heatmap, overlay_heatmap_on_image, tensor_image_to_numpy
from src.explainability.saliency_export import save_overlay_image, save_saliency_array
from src.explainability.target_layers import resolve_target_layer
from src.models.factory import build_model
from src.utils.checkpoint import load_model_weights
from src.utils.config import load_yaml_config
from src.utils.device import get_device
from src.utils.io import read_csv
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

logger = get_logger(__name__)


def run_explainability_experiment(
    dataset_name: str,
    model_name: str = "maizeformerx",
    checkpoint_path: str | Path | None = None,
    aug_config_name: str | None = None,
    max_cases: int = 12,
    target_layer_path: str | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    train_cfg = load_yaml_config(TRAIN_CONFIG_DIR / f"{dataset_name}.yaml")
    class_map = build_class_map(dataset_name)

    if aug_config_name is None:
        aug_config_name = train_cfg["augmentation"]["config_name"]
    aug_cfg = load_yaml_config(Path("configs/aug") / f"{aug_config_name}.yaml")
    model_cfg = load_yaml_config(MODEL_CONFIG_PATHS[model_name])

    device = get_device(device_name)
    seed_everything(42)

    model = build_model(model_name=model_name, model_cfg=model_cfg, num_classes=len(class_map.class_to_index))
    model.to(device)
    if checkpoint_path is not None:
        load_model_weights(checkpoint_path, model, map_location=device, strict=True)

    transforms = build_train_val_test_transforms(aug_cfg)
    test_records = read_csv(Path("data/interim/split_files") / f"{dataset_name}_test.csv")

    test_loader = build_manifest_dataloader(
        records=test_records,
        transform=transforms["test"],
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        shuffle=False,
        weighted_sampling=False,
        drop_last=False,
        return_metadata=True,
    )

    target_layer = resolve_target_layer(model, model_name=model_name, layer_path=target_layer_path)
    gradcam = GradCAM(model=model, target_layer=target_layer, device=device)

    output_root = OUTPUT_SALIENCY_DIR / dataset_name / model_name
    entries = []
    processed = 0
    class_names = list(class_map.class_to_index.keys())

    for batch in test_loader:
        inputs, targets, metadata = batch
        result = gradcam.generate(inputs, target_indices=targets)

        image_np = tensor_image_to_numpy(inputs[0], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        cam_np = result.saliency[0, 0].numpy()
        heatmap_np = cam_to_heatmap(cam_np)
        overlay_np = overlay_heatmap_on_image(image_np, heatmap_np, alpha=0.4)

        true_idx = int(targets[0].item())
        pred_idx = int(result.predicted_indices[0].item())

        file_stem = Path(metadata["file_name"][0]).stem if isinstance(metadata["file_name"], list) else Path(metadata["file_name"]).stem
        saliency_path = output_root / f"{file_stem}_saliency.npy"
        overlay_path = output_root / f"{file_stem}_overlay.png"

        save_saliency_array(saliency_path, cam_np)
        save_overlay_image(overlay_path, overlay_np)

        probability = float(torch.softmax(result.logits, dim=1)[0, pred_idx].item())
        entry = build_case_entry(
            image_path=metadata["image_path"][0] if isinstance(metadata["image_path"], list) else metadata["image_path"],
            true_label=class_names[true_idx],
            predicted_label=class_names[pred_idx],
            saliency_path=str(saliency_path),
            overlay_path=str(overlay_path),
            probability=probability,
            metadata={"file_name": metadata["file_name"][0] if isinstance(metadata["file_name"], list) else metadata["file_name"]},
        )
        entries.append(entry)

        processed += 1
        if processed >= max_cases:
            break

    gradcam.remove_hooks()

    casebook_path = output_root / "casebook.json"
    save_casebook(casebook_path, entries)

    logger.info("Saved explainability casebook to %s", casebook_path)
    return {
        "dataset": dataset_name,
        "model": model_name,
        "num_cases": len(entries),
        "casebook_path": str(casebook_path),
        "cases": entries,
    }