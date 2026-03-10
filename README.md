````markdown
# MaizeFormerX: Lightweight Multi-Scale Vision Transformer for Maize Disease Classification
MaizeFormerX is a deep learning framework for maize leaf disease classification using lightweight vision models, centered on a custom **multi-scale vision transformer** architecture. The repository implements the **full experimental pipeline** required to reproduce and extend the study, including dataset preparation, manifest generation, stratified splitting, model training, in-domain benchmarking, augmentation analysis, cross-domain evaluation, ablation studies, explainability, efficiency profiling, and statistical comparison. The framework is designed for **reproducible experimentation** across multiple maize image datasets with different class taxonomies and acquisition conditions. In addition to the proposed **MaizeFormerX** model, the repository supports several compact transformer and CNN baselines under a unified training and evaluation protocol. The codebase also includes **Grad-CAM-based explainability utilities** and a lightweight **FastAPI serving layer** for inference and demonstration.

---

## Table of Contents

- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Method Overview](#method-overview)
- [Experimental Scope](#experimental-scope)
- [Supported Datasets](#supported-datasets)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Configuration Guide](#configuration-guide)
- [Training](#training)
- [Evaluation](#evaluation)
- [Full Experiment Pipelines](#full-experiment-pipelines)
- [Explainability](#explainability)
- [Profiling and Efficiency](#profiling-and-efficiency)
- [Testing](#testing)
- [Serving API](#serving-api)
- [Reproducibility Notes](#reproducibility-notes)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Key Features

- Lightweight multi-scale transformer architecture for maize disease classification
- Multi-scale patch embedding for local and global symptom representation
- Cross-Scale Attention Fusion (CSAF) for integrating multi-resolution token streams
- Unified training pipeline for transformer and CNN baselines
- Independent dataset preparation with manifest generation and stratified splits
- In-domain benchmarking across multiple datasets and augmentation settings
- Cross-domain evaluation under a shared-label protocol
- Architecture ablation support for component-level analysis
- Grad-CAM explainability pipeline with overlay export and casebook generation
- Efficiency profiling including parameters, FLOPs, CPU latency, GPU latency, and memory
- FastAPI serving layer for inference and demo deployment
- Pytest-based testing suite for core modules

---

## Repository Structure

```text
maizeformerx/
├── requirements.txt
├── setup.py
├── configs/
├── data/
├── src/
├── scripts/
├── outputs/
````

### Main Directories

* `configs/` contains YAML configuration files for training, models, experiments, and augmentation policies.
* `data/` stores metadata.
* `src/` contains the full implementation of models, data handling, training, evaluation, explainability, profiling, serving, and CLI tools.
* `scripts/` provides shell scripts for dataset preparation and full experiment execution.
* `outputs/` stores checkpoints, logs, metrics, tables, figures, saliency maps, profiles, and reports.

---

## Method Overview

The repository implements the complete **MaizeFormerX** pipeline as follows:

1. Raw maize leaf images are organized into class-specific directories.
2. Dataset manifests are generated automatically from the raw folder structure.
3. Fixed stratified train/validation/test splits are created and saved.
4. Images are resized and normalized, and dataset-specific augmentation policies are applied during training.
5. MaizeFormerX performs multi-scale patch embedding using multiple patch sizes and strides.
6. Cross-Scale Attention Fusion (CSAF) projects and fuses the multi-scale token streams into a shared representation.
7. A transformer encoder refines the fused representation.
8. A classification head produces disease logits.
9. Performance is evaluated using metrics such as Accuracy, Micro-F1, Macro-F1, PR-AUC, and MCC.
10. Explainability is supported via Grad-CAM overlays and casebook export.
11. Deployment-oriented measurements are computed through efficiency profiling.

The framework also supports baseline comparisons against lightweight ViTs and compact CNNs under the same data pipeline and evaluation setup.

---

## Experimental Scope

The repository is structured to reproduce and extend the following experiment categories.

### In-domain benchmark

Each model is trained and evaluated on the same dataset using fixed train/validation/test splits.

### Augmentation sweep

Performance is compared under multiple augmentation expansion settings such as ×2, ×4, and ×6.

### Cross-domain evaluation

Models trained on one dataset are evaluated on another dataset using a shared-label protocol without target fine-tuning.

### Ablation study

Key components of MaizeFormerX can be disabled to quantify their contribution:

* without CSAF
* without multi-scale embedding
* without transformer encoder
* full model

### Explainability

Grad-CAM is used to generate saliency maps and overlays for qualitative auditing of predictions.

### Efficiency profiling

The framework supports:

* parameter counting
* FLOP estimation
* CPU latency
* GPU latency
* peak memory measurement

### Statistical comparison

Seed-wise outputs can be compared using paired statistical tests and effect sizes.

---

## Supported Datasets
Three datasets were used:

* Dataverse (Role: in-domain, augmentation, cross-domain)
* Tanzania (Role: in-domain, augmentation, cross-domain)
* Plagues Maiz (Role: in-domain, augmentation, cross-domain, fine-grained analysis)


Each dataset should be organized using a folder-per-class structure:

```text
data/raw/<dataset_name>/
├── class_1/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── class_2/
│   ├── img_003.jpg
│   └── ...
└── ...
```

The repository expects raw datasets to be placed manually under the `data/raw/` directory before running preparation scripts.

---

## Installation

### Requirements

* Python 3.10 or newer
* PyTorch and torchvision compatible with your system
* Recommended: CUDA-enabled GPU for training and profiling

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/rezaul-h/MaizeFormerX.git
cd maizeformerx
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For development and testing:

```bash
pip install -e .[dev]
```

---

## Data Preparation

Before training, prepare dataset artifacts including:

* manifest CSV files
* split CSV files
* runtime label maps

### Prepare Dataverse

```bash
bash scripts/prepare_dataverse.sh ./data/raw/dataverse
```

### Prepare Tanzania

```bash
bash scripts/prepare_tanzania.sh ./data/raw/tanzania
```

### Prepare Plagues Maiz

```bash
bash scripts/prepare_plagues_maiz.sh ./data/raw/plagues_maiz
```

### Direct CLI Alternative

```bash
python -m src.cli.prepare_data --dataset dataverse --dataset-root ./data/raw/dataverse --seed 42
python -m src.cli.prepare_data --dataset tanzania --dataset-root ./data/raw/tanzania --seed 42
python -m src.cli.prepare_data --dataset plagues_maiz --dataset-root ./data/raw/plagues_maiz --seed 42
```

Generated artifacts are stored under:

```text
data/interim/manifests/
data/interim/split_files/
data/interim/label_maps/
```

---

## Configuration Guide

The project uses YAML-based configuration throughout the pipeline.

### Global configuration

`configs/global.yaml` defines global runtime, logging, reproducibility, and evaluation settings.

### Path configuration

`configs/paths.yaml` defines canonical project directories.

### Training configuration

`configs/train/*.yaml` defines dataset-specific training settings such as:

* batch size
* optimizer
* scheduler
* epochs
* early stopping
* augmentation config binding

### Model configuration

`configs/model/*.yaml` defines model-specific architecture settings.

### Experiment configuration

`configs/experiment/*.yaml` defines experiment-level orchestration for:

* in-domain benchmark
* augmentation sweep
* cross-domain evaluation
* ablation study
* explainability
* efficiency
* statistics

### Augmentation configuration

`configs/aug/*.yaml` defines dataset-specific augmentation recipes for each expansion factor.

This design allows most experiments to be modified without editing Python code.

---

## Training

A single training run can be launched through the training CLI.

### Example: Train MaizeFormerX on Dataverse

```bash
python -m src.cli.train \
  --dataset dataverse \
  --model maizeformerx \
  --seed 42 \
  --device auto \
  --checkpoint-dir outputs/checkpoints/dataverse_maizeformerx_seed42
```

### Example: Train EfficientNet-B1 on Tanzania with a specific augmentation config

```bash
python -m src.cli.train \
  --dataset tanzania \
  --model efficientnet_b1 \
  --aug-config tanzania_x2 \
  --seed 42 \
  --device auto \
  --checkpoint-dir outputs/checkpoints/tanzania_efficientnetb1_seed42
```

Best and last checkpoints are stored in the specified checkpoint directory.

---

## Evaluation

A trained checkpoint can be evaluated on any split.

### Example: Evaluate on the test split

```bash
python -m src.cli.evaluate \
  --dataset dataverse \
  --model maizeformerx \
  --checkpoint outputs/checkpoints/dataverse_maizeformerx_seed42/best.pt \
  --split test \
  --device auto \
  --output-json outputs/reports/dataverse_eval.json
```

The evaluation output includes:

* scalar metrics
* confusion matrix
* per-class metrics
* logits
* probabilities
* predictions
* targets

---

## Full Experiment Pipelines

### Run full in-domain benchmark

```bash
bash scripts/run_all_in_domain.sh
```

### Run full cross-domain benchmark

```bash
bash scripts/run_all_cross_domain.sh
```

### Run full ablation study

```bash
bash scripts/run_all_ablations.sh
```

### Run full efficiency profiling

```bash
bash scripts/run_all_efficiency.sh
```

### Build final experiment tables

```bash
bash scripts/make_final_tables.sh
```

### Generic reproduction entry point

```bash
python -m src.cli.reproduce --experiment in_domain --device auto
python -m src.cli.reproduce --experiment cross_domain --device auto
python -m src.cli.reproduce --experiment ablation --device auto
python -m src.cli.reproduce --experiment efficiency --device auto
python -m src.cli.reproduce --experiment statistics --device auto
```

---

## Explainability

The project includes Grad-CAM-based explainability for trained models.

### Generate explainability outputs

```bash
python -m src.cli.explain \
  --dataset dataverse \
  --model maizeformerx \
  --checkpoint outputs/checkpoints/dataverse_maizeformerx_seed42/best.pt \
  --device auto \
  --max-cases 12
```

### Outputs

Explainability artifacts are stored under:

```text
outputs/saliency/<dataset>/<model>/
```

This includes:

* raw saliency arrays
* overlay PNG files
* casebook JSON metadata

### Batch shell script

```bash
bash scripts/run_all_xai.sh dataverse maizeformerx outputs/checkpoints/dataverse_maizeformerx_seed42/best.pt
```

---

## Profiling and Efficiency

The repository supports model profiling at both the CLI and experiment level.

### Single-model profiling

```bash
python -m src.cli.profile \
  --model maizeformerx \
  --num-classes 3 \
  --output-json outputs/profiles/maizeformerx_profile.json
```

This can report:

* total parameters
* trainable parameters
* FLOPs
* CPU latency
* GPU latency
* peak GPU memory

### Full efficiency experiment

```bash
bash scripts/run_all_efficiency.sh
```

---

## Output Directory Guide

Generated artifacts are organized as follows:

```text
outputs/
├── checkpoints/   # saved model checkpoints
├── logs/          # runtime logs
├── metrics/       # experiment result JSON files
├── predictions/   # optional prediction exports
├── tables/        # CSV summary tables
├── figures/       # plots and publication figures
├── saliency/      # Grad-CAM maps and overlays
├── profiles/      # profiling outputs
└── reports/       # statistical and evaluation reports
```

---

## Testing

The repository includes a lightweight test suite for critical components.

### Run all tests

```bash
pytest tests -q
```

### Coverage example

```bash
pytest tests --cov=src --cov-report=term-missing
```

The current tests cover:

* split generation
* shared-label protocol
* augmentation building
* model forward passes
* CSAF output shape
* metrics
* Grad-CAM
* reproducibility

---

## Serving API

A lightweight FastAPI service is included for inference and demo deployment.

### Launch the API

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

### Health check

`GET /health`

### Load a model

`POST /load_model`

Example payload:

```json
{
  "model_name": "maizeformerx",
  "dataset_name": "dataverse",
  "checkpoint_path": "outputs/checkpoints/dataverse_maizeformerx_seed42/best.pt",
  "device": "auto"
}
```

### Predict from an uploaded image

`POST /predict`

### Get explainability overlay

`POST /explain/overlay`

This makes the repository directly usable for lightweight model serving and demonstrations.

---

## Reproducibility Notes

The repository has been designed to support reproducible experimentation through the following mechanisms:

* fixed and saved train/validation/test splits
* metadata-driven class maps
* configuration-driven experiments
* deterministic seeding utilities
* multi-seed evaluation support
* explicit output artifact tracking
* modular experiment scripts for controlled reruns

For strongest reproducibility, keep the following fixed across runs:

* raw dataset version
* split files
* dependency versions
* hardware environment when profiling latency or memory

---

## License

```text
This project is released under the MIT License.
```

---

## Acknowledgments

This framework builds on the broader open-source deep learning ecosystem, including:

* PyTorch
* torchvision
* timm
* scikit-learn
* FastAPI
* matplotlib

```
```
