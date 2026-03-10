"""
CLI for model profiling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.constants import MODEL_CONFIG_PATHS
from src.models.factory import build_model
from src.profiling.latency_cpu import measure_cpu_latency
from src.profiling.latency_gpu import measure_gpu_latency
from src.profiling.memory import measure_peak_gpu_memory
from src.profiling.params_flops import (
    count_total_parameters,
    count_trainable_parameters,
    estimate_flops_with_thop,
)
from src.utils.config import load_yaml_config
from src.utils.io import write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile a model.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--output-json", type=str, default="outputs/profiles/profile_result.json")
    return parser


def main(args=None) -> int:
    if args is None or not hasattr(args, "model"):
        parser = build_parser()
        args = parser.parse_args()

    model_cfg = load_yaml_config(MODEL_CONFIG_PATHS[args.model])
    model = build_model(args.model, model_cfg, num_classes=args.num_classes)

    result = {
        "model": args.model,
        "total_parameters": count_total_parameters(model),
        "trainable_parameters": count_trainable_parameters(model),
    }
    result.update(estimate_flops_with_thop(model, input_size=(1, 3, 224, 224)))
    result.update(measure_cpu_latency(model, input_size=(1, 3, 224, 224)))
    result.update(measure_gpu_latency(model, input_size=(1, 3, 224, 224)))
    result.update(measure_peak_gpu_memory(model, input_size=(1, 3, 224, 224)))

    write_json(args.output_json, result)
    logger.info("Saved profiling results to %s", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())