"""
CLI for explainability generation.
"""

from __future__ import annotations

import argparse

from src.experiments.run_explainability import run_explainability_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanations.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="maizeformerx")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--aug-config", type=str, default=None)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--target-layer", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main(args=None) -> int:
    if args is None or not hasattr(args, "dataset"):
        parser = build_parser()
        args = parser.parse_args()

    run_explainability_experiment(
        dataset_name=args.dataset,
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        aug_config_name=args.aug_config,
        max_cases=args.max_cases,
        target_layer_path=args.target_layer,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())