"""
Explainability exports.
"""

from src.explainability.casebook import build_case_entry, save_casebook
from src.explainability.gradcam import GradCAM, GradCAMResult
from src.explainability.overlays import (
    cam_to_heatmap,
    overlay_heatmap_on_image,
    save_numpy_image,
    tensor_image_to_numpy,
)
from src.explainability.saliency_export import (
    save_explainability_metadata,
    save_overlay_image,
    save_saliency_array,
)
from src.explainability.target_layers import (
    get_default_target_layer,
    resolve_layer_path,
    resolve_target_layer,
)

__all__ = [
    "GradCAM",
    "GradCAMResult",
    "build_case_entry",
    "cam_to_heatmap",
    "get_default_target_layer",
    "overlay_heatmap_on_image",
    "resolve_layer_path",
    "resolve_target_layer",
    "save_casebook",
    "save_explainability_metadata",
    "save_numpy_image",
    "save_overlay_image",
    "save_saliency_array",
    "tensor_image_to_numpy",
]