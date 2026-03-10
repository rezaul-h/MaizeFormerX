"""
Main entry point for the MaizeFormerX project.

This module provides a simple top-level command-line interface that routes
execution to the appropriate workflow. It is intentionally lightweight and
serves as the stable front door for the repository.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

from src.constants import (
    PROJECT_NAME,
    PROJECT_VERSION,
    SUPPORTED_EXPERIMENTS,
    SUPPORTED_TASKS,
    TASK_EVALUATE,
    TASK_EXPLAIN,
    TASK_PREPARE_DATA,
    TASK_PROFILE,
    TASK_REPRODUCE,
    TASK_TRAIN,
)


def _build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog=PROJECT_NAME,
        description="Unified command-line interface for MaizeFormerX workflows.",
    )

    parser.add_argument(
        "task",
        choices=SUPPORTED_TASKS,
        help="Top-level task to execute.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a primary YAML config file.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=SUPPORTED_EXPERIMENTS,
        help="Optional experiment type for task-specific workflows.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset name.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed override.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-friendly execution behavior.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {PROJECT_VERSION}",
    )

    return parser


def _run_prepare_data(args: argparse.Namespace) -> int:
    """Dispatch the data preparation task."""
    try:
        from src.cli.prepare_data import main as prepare_data_main
    except ImportError as exc:
        print(f"[ERROR] Failed to import prepare_data CLI: {exc}", file=sys.stderr)
        return 1

    return int(prepare_data_main(args=args) or 0)


def _run_train(args: argparse.Namespace) -> int:
    """Dispatch the training task."""
    try:
        from src.cli.train import main as train_main
    except ImportError as exc:
        print(f"[ERROR] Failed to import train CLI: {exc}", file=sys.stderr)
        return 1

    return int(train_main(args=args) or 0)


def _run_evaluate(args: argparse.Namespace) -> int:
    """Dispatch the evaluation task."""
    try:
        from src.cli.evaluate import main as evaluate_main
    except ImportError as exc:
        print(f"[ERROR] Failed to import evaluate CLI: {exc}", file=sys.stderr)
        return 1

    return int(evaluate_main(args=args) or 0)


def _run_explain(args: argparse.Namespace) -> int:
    """Dispatch the explainability task."""
    try:
        from src.cli.explain import main as explain_main
    except ImportError as exc:
        print(f"[ERROR] Failed to import explain CLI: {exc}", file=sys.stderr)
        return 1

    return int(explain_main(args=args) or 0)


def _run_profile(args: argparse.Namespace) -> int:
    """Dispatch the profiling task."""
    try:
        from src.cli.profile import main as profile_main
    except ImportError as exc:
        print(f"[ERROR] Failed to import profile CLI: {exc}", file=sys.stderr)
        return 1

    return int(profile_main(args=args) or 0)


def _run_reproduce(args: argparse.Namespace) -> int:
    """Dispatch the reproduction task."""
    try:
        from src.cli.reproduce import main as reproduce_main
    except ImportError as exc:
        print(f"[ERROR] Failed to import reproduce CLI: {exc}", file=sys.stderr)
        return 1

    return int(reproduce_main(args=args) or 0)


def _task_dispatch_table() -> dict[str, Callable[[argparse.Namespace], int]]:
    """Return the mapping from task name to dispatcher."""
    return {
        TASK_PREPARE_DATA: _run_prepare_data,
        TASK_TRAIN: _run_train,
        TASK_EVALUATE: _run_evaluate,
        TASK_EXPLAIN: _run_explain,
        TASK_PROFILE: _run_profile,
        TASK_REPRODUCE: _run_reproduce,
    }


def main(argv: list[str] | None = None) -> int:
    """
    Execute the main CLI.

    Parameters
    ----------
    argv:
        Optional CLI argument list. If None, argparse reads from sys.argv.

    Returns
    -------
    int
        Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch_table = _task_dispatch_table()
    task_runner = dispatch_table.get(args.task)

    if task_runner is None:
        print(
            f"[ERROR] Unsupported task '{args.task}'. Supported tasks: {SUPPORTED_TASKS}",
            file=sys.stderr,
        )
        return 1

    return task_runner(args)


if __name__ == "__main__":
    raise SystemExit(main())