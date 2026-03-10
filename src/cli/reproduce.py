"""
CLI for reproducing full experiments.
"""

from __future__ import annotations

import argparse

from src.experiments.orchestrator import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a predefined experiment pipeline.")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=[
            "in_domain",
            "augmentation_sweep",
            "cross_domain",
            "ablation",
            "explainability",
            "efficiency",
            "statistics",
        ],
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main(args=None) -> int:
    if args is None or not hasattr(args, "experiment"):
        parser = build_parser()
        args = parser.parse_args()

    kwargs = {}
    if args.config is not None:
        kwargs["experiment_config_path"] = args.config

    if args.experiment == "explainability":
        raise ValueError(
            "The explainability experiment requires dataset/model/checkpoint-specific arguments. "
            "Use src/cli/explain.py instead."
        )

    run_experiment(args.experiment, device_name=args.device, **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())