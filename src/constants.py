"""
Project-wide constants for MaizeFormerX.

Keep this file limited to static, reusable constants that are needed across
multiple modules. Avoid placing runtime configuration here; use YAML configs
for experiment-specific settings.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------

PROJECT_NAME: str = "maizeformerx"
PROJECT_VERSION: str = "0.1.0"
PROJECT_DESCRIPTION: str = (
    "MaizeFormerX: lightweight multi-scale vision transformer for maize disease classification"
)

# ---------------------------------------------------------------------
# Core paths
# ---------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
SRC_DIR: Path = REPO_ROOT / "src"
CONFIGS_DIR: Path = REPO_ROOT / "configs"
DATA_DIR: Path = REPO_ROOT / "data"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
TESTS_DIR: Path = REPO_ROOT / "tests"
NOTEBOOKS_DIR: Path = REPO_ROOT / "notebooks"

RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
METADATA_DIR: Path = DATA_DIR / "metadata"

# ---------------------------------------------------------------------
# Dataset names
# ---------------------------------------------------------------------

DATASET_DATAVERSE: str = "dataverse"
DATASET_TANZANIA: str = "tanzania"
DATASET_PLAGUES_MAIZ: str = "plagues_maiz"

SUPPORTED_DATASETS: tuple[str, ...] = (
    DATASET_DATAVERSE,
    DATASET_TANZANIA,
    DATASET_PLAGUES_MAIZ,
)

DATASET_CLASS_FILES: dict[str, Path] = {
    DATASET_DATAVERSE: METADATA_DIR / "dataverse_classes.json",
    DATASET_TANZANIA: METADATA_DIR / "tanzania_classes.json",
    DATASET_PLAGUES_MAIZ: METADATA_DIR / "plagues_maiz_classes.json",
}

DATASET_RAW_DIRS: dict[str, Path] = {
    DATASET_DATAVERSE: RAW_DATA_DIR / DATASET_DATAVERSE,
    DATASET_TANZANIA: RAW_DATA_DIR / DATASET_TANZANIA,
    DATASET_PLAGUES_MAIZ: RAW_DATA_DIR / DATASET_PLAGUES_MAIZ,
}

DATASET_PROCESSED_DIRS: dict[str, Path] = {
    DATASET_DATAVERSE: PROCESSED_DATA_DIR / DATASET_DATAVERSE,
    DATASET_TANZANIA: PROCESSED_DATA_DIR / DATASET_TANZANIA,
    DATASET_PLAGUES_MAIZ: PROCESSED_DATA_DIR / DATASET_PLAGUES_MAIZ,
}

# ---------------------------------------------------------------------
# Label and split artifacts
# ---------------------------------------------------------------------

DATASET_SUMMARY_CSV: Path = METADATA_DIR / "dataset_summary.csv"
SHARED_LABEL_MAPS_JSON: Path = METADATA_DIR / "shared_label_maps.json"

MANIFESTS_DIR: Path = INTERIM_DATA_DIR / "manifests"
SPLIT_FILES_DIR: Path = INTERIM_DATA_DIR / "split_files"
LABEL_MAPS_DIR: Path = INTERIM_DATA_DIR / "label_maps"

DEFAULT_SPLIT_RATIOS: dict[str, float] = {
    "train": 0.80,
    "val": 0.05,
    "test": 0.15,
}

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

# ---------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------

MODEL_MAIZEFORMERX: str = "maizeformerx"
MODEL_MOBILEVIT: str = "mobilevit"
MODEL_EFFICIENTFORMER: str = "efficientformer"
MODEL_TINYVIT: str = "tinyvit"
MODEL_SWINV2: str = "swinv2"
MODEL_SHUFFLENETV2: str = "shufflenetv2"
MODEL_GHOSTNET: str = "ghostnet"
MODEL_MOBILENETV3: str = "mobilenetv3"
MODEL_EFFICIENTNET_B0: str = "efficientnet_b0"
MODEL_EFFICIENTNET_B1: str = "efficientnet_b1"
MODEL_MIXMOBILENET: str = "mixmobilenet"

SUPPORTED_MODELS: tuple[str, ...] = (
    MODEL_MAIZEFORMERX,
    MODEL_MOBILEVIT,
    MODEL_EFFICIENTFORMER,
    MODEL_TINYVIT,
    MODEL_SWINV2,
    MODEL_SHUFFLENETV2,
    MODEL_GHOSTNET,
    MODEL_MOBILENETV3,
    MODEL_EFFICIENTNET_B0,
    MODEL_EFFICIENTNET_B1,
    MODEL_MIXMOBILENET,
)

VIT_MODELS: tuple[str, ...] = (
    MODEL_MAIZEFORMERX,
    MODEL_MOBILEVIT,
    MODEL_EFFICIENTFORMER,
    MODEL_TINYVIT,
    MODEL_SWINV2,
)

CNN_MODELS: tuple[str, ...] = (
    MODEL_SHUFFLENETV2,
    MODEL_GHOSTNET,
    MODEL_MOBILENETV3,
    MODEL_EFFICIENTNET_B0,
    MODEL_EFFICIENTNET_B1,
    MODEL_MIXMOBILENET,
)

# ---------------------------------------------------------------------
# Experiment names
# ---------------------------------------------------------------------

EXPERIMENT_IN_DOMAIN: str = "in_domain"
EXPERIMENT_AUGMENTATION_SWEEP: str = "augmentation_sweep"
EXPERIMENT_CROSS_DOMAIN: str = "cross_domain"
EXPERIMENT_ABLATION: str = "ablation"
EXPERIMENT_EXPLAINABILITY: str = "explainability"
EXPERIMENT_EFFICIENCY: str = "efficiency"
EXPERIMENT_STATISTICS: str = "statistics"

SUPPORTED_EXPERIMENTS: tuple[str, ...] = (
    EXPERIMENT_IN_DOMAIN,
    EXPERIMENT_AUGMENTATION_SWEEP,
    EXPERIMENT_CROSS_DOMAIN,
    EXPERIMENT_ABLATION,
    EXPERIMENT_EXPLAINABILITY,
    EXPERIMENT_EFFICIENCY,
    EXPERIMENT_STATISTICS,
)

# ---------------------------------------------------------------------
# File extensions and image handling
# ---------------------------------------------------------------------

IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

DEFAULT_IMAGE_SIZE: int = 224
DEFAULT_IN_CHANNELS: int = 3
DEFAULT_COLOR_MODE: str = "rgb"

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------
# Reproducibility and training defaults
# ---------------------------------------------------------------------

DEFAULT_SEED: int = 42
DEFAULT_MULTI_SEEDS: tuple[int, ...] = (42, 52, 62, 72, 82)

DEFAULT_BATCH_SIZE: int = 32
DEFAULT_NUM_WORKERS: int = 8
DEFAULT_MAX_EPOCHS: int = 30

DEFAULT_OPTIMIZER: str = "adam"
DEFAULT_LEARNING_RATE: float = 5e-5
DEFAULT_WEIGHT_DECAY: float = 5e-5
DEFAULT_DROPOUT: float = 0.30

# ---------------------------------------------------------------------
# Evaluation and monitoring
# ---------------------------------------------------------------------

PRIMARY_METRIC: str = "accuracy"
MONITOR_MODE_MAX: str = "max"
MONITOR_MODE_MIN: str = "min"

SUPPORTED_METRICS: tuple[str, ...] = (
    "loss",
    "accuracy",
    "micro_f1",
    "macro_f1",
    "pr_auc",
    "mcc",
)

# ---------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------

OUTPUT_LOGS_DIR: Path = OUTPUTS_DIR / "logs"
OUTPUT_CHECKPOINTS_DIR: Path = OUTPUTS_DIR / "checkpoints"
OUTPUT_PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"
OUTPUT_METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
OUTPUT_TABLES_DIR: Path = OUTPUTS_DIR / "tables"
OUTPUT_FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
OUTPUT_SALIENCY_DIR: Path = OUTPUTS_DIR / "saliency"
OUTPUT_PROFILES_DIR: Path = OUTPUTS_DIR / "profiles"
OUTPUT_REPORTS_DIR: Path = OUTPUTS_DIR / "reports"

# ---------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------

GLOBAL_CONFIG_PATH: Path = CONFIGS_DIR / "global.yaml"
PATHS_CONFIG_PATH: Path = CONFIGS_DIR / "paths.yaml"

TRAIN_CONFIG_DIR: Path = CONFIGS_DIR / "train"
MODEL_CONFIG_DIR: Path = CONFIGS_DIR / "model"
EXPERIMENT_CONFIG_DIR: Path = CONFIGS_DIR / "experiment"
AUG_CONFIG_DIR: Path = CONFIGS_DIR / "aug"

DEFAULT_TRAIN_CONFIG_PATH: Path = TRAIN_CONFIG_DIR / "default.yaml"

MODEL_CONFIG_PATHS: dict[str, Path] = {
    MODEL_MAIZEFORMERX: MODEL_CONFIG_DIR / "maizeformerx.yaml",
    MODEL_MOBILEVIT: MODEL_CONFIG_DIR / "mobilevit.yaml",
    MODEL_EFFICIENTFORMER: MODEL_CONFIG_DIR / "efficientformer.yaml",
    MODEL_TINYVIT: MODEL_CONFIG_DIR / "tinyvit.yaml",
    MODEL_SWINV2: MODEL_CONFIG_DIR / "swinv2.yaml",
    MODEL_SHUFFLENETV2: MODEL_CONFIG_DIR / "shufflenetv2.yaml",
    MODEL_GHOSTNET: MODEL_CONFIG_DIR / "ghostnet.yaml",
    MODEL_MOBILENETV3: MODEL_CONFIG_DIR / "mobilenetv3.yaml",
    MODEL_EFFICIENTNET_B0: MODEL_CONFIG_DIR / "efficientnet_b0.yaml",
    MODEL_EFFICIENTNET_B1: MODEL_CONFIG_DIR / "efficientnet_b1.yaml",
    MODEL_MIXMOBILENET: MODEL_CONFIG_DIR / "mixmobilenet.yaml",
}

EXPERIMENT_CONFIG_PATHS: dict[str, Path] = {
    EXPERIMENT_IN_DOMAIN: EXPERIMENT_CONFIG_DIR / "in_domain.yaml",
    EXPERIMENT_AUGMENTATION_SWEEP: EXPERIMENT_CONFIG_DIR / "augmentation_sweep.yaml",
    EXPERIMENT_CROSS_DOMAIN: EXPERIMENT_CONFIG_DIR / "cross_domain.yaml",
    EXPERIMENT_ABLATION: EXPERIMENT_CONFIG_DIR / "ablation.yaml",
    EXPERIMENT_EXPLAINABILITY: EXPERIMENT_CONFIG_DIR / "explainability.yaml",
    EXPERIMENT_EFFICIENCY: EXPERIMENT_CONFIG_DIR / "efficiency.yaml",
    EXPERIMENT_STATISTICS: EXPERIMENT_CONFIG_DIR / "statistics.yaml",
}

# ---------------------------------------------------------------------
# Runtime task registry names
# ---------------------------------------------------------------------

TASK_PREPARE_DATA: str = "prepare_data"
TASK_TRAIN: str = "train"
TASK_EVALUATE: str = "evaluate"
TASK_EXPLAIN: str = "explain"
TASK_PROFILE: str = "profile"
TASK_REPRODUCE: str = "reproduce"

SUPPORTED_TASKS: tuple[str, ...] = (
    TASK_PREPARE_DATA,
    TASK_TRAIN,
    TASK_EVALUATE,
    TASK_EXPLAIN,
    TASK_PROFILE,
    TASK_REPRODUCE,
)