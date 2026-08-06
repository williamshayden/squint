"""Standard-library command line interface for checkpoint runs and comparison."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from math import isfinite
from pathlib import Path
from typing import NoReturn, cast

from edge_perception.compare import compare_runs
from edge_perception.config import RunConfig, load_run_config
from edge_perception.contracts import Region
from edge_perception.detectors.registry import load_detector
from edge_perception.runner import run_checkpoint, validate_output_directory


class _CliError(ValueError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _CliError(message)


def _crop(value: str) -> Region:
    try:
        region_id, coordinates = value.split(":", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("crop must use ID:X,Y,WIDTH,HEIGHT") from error
    if not region_id:
        raise argparse.ArgumentTypeError("crop ID must not be empty")
    parts = coordinates.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must use ID:X,Y,WIDTH,HEIGHT")
    try:
        x, y, width, height = (int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("crop coordinates must be integers") from error
    if x < 0 or y < 0:
        raise argparse.ArgumentTypeError("crop origin must not be negative")
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("crop width and height must be positive")
    return Region(region_id, x, y, width, height)


def _threshold(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("threshold must be a number between 0 and 1") from error
    if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return threshold


def _nonnegative(name: str) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            count = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from error
        if count < 0:
            raise argparse.ArgumentTypeError(f"{name} must not be negative")
        return count

    return parse


def _nonnegative_tolerance(name: str) -> Callable[[str], float]:
    def parse(value: str) -> float:
        try:
            tolerance = float(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{name} must be a number") from error
        if not isfinite(tolerance) or tolerance < 0.0:
            raise argparse.ArgumentTypeError(f"{name} must be finite and non-negative")
        return tolerance

    return parse


def _build_parser() -> _Parser:
    parser = _Parser(prog="edge-perception", add_help=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a chronological video checkpoint")
    run.add_argument("input", type=Path, nargs="?")
    run.add_argument("--config", type=Path)
    run.add_argument("--output", type=Path, default=argparse.SUPPRESS)
    run.add_argument("--crop", action="append", type=_crop, default=argparse.SUPPRESS)
    run.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default=argparse.SUPPRESS
    )
    run.add_argument("--threshold", type=_threshold, default=argparse.SUPPRESS)
    run.add_argument(
        "--warmup-runs", type=_nonnegative("warmup-runs"), default=argparse.SUPPRESS
    )
    run.add_argument("--max-frames", type=_nonnegative("max-frames"), default=argparse.SUPPRESS)
    run.add_argument(
        "--annotate-every", type=_nonnegative("annotate-every"), default=argparse.SUPPRESS
    )

    compare = subparsers.add_parser("compare", help="compare semantic checkpoint detections")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--box-atol", type=_nonnegative_tolerance("box-atol"), default=0.01)
    compare.add_argument(
        "--score-atol",
        type=_nonnegative_tolerance("score-atol"),
        default=1e-4,
    )
    return parser


def _run_command(args: argparse.Namespace) -> int:
    config = _resolve_run_config(args)
    if not config.input_path.is_file():
        raise _CliError(f"input video does not exist: {config.input_path}")
    validate_output_directory(config.output_dir)
    detector = load_detector(
        config.detector_id,
        threshold=config.threshold,
        device=config.device,
    )
    summary = run_checkpoint(config, detector)
    print(f"output: {config.output_dir}")
    print(
        f"status={summary.get('status')} "
        f"frames={summary.get('frames_processed')} "
        f"inferences={summary.get('inference_count')}"
    )
    return 0


def _resolve_run_config(args: argparse.Namespace) -> RunConfig:
    input_path = cast(Path | None, args.input)
    config_path = cast(Path | None, args.config)
    if input_path is None and config_path is None:
        raise _CliError("run requires INPUT or --config")
    if input_path is not None and config_path is not None:
        raise _CliError("INPUT cannot be combined with --config")
    if config_path is None:
        if not hasattr(args, "output"):
            raise _CliError("--output is required without --config")
        return _explicit_run_config(args, input_path)
    return _apply_config_overrides(load_run_config(config_path), args)


def _explicit_run_config(args: argparse.Namespace, input_path: Path | None) -> RunConfig:
    if input_path is None:
        raise AssertionError("input path is required for an explicit run")
    output_dir = cast(Path, args.output)
    if output_dir.resolve() == input_path.resolve():
        raise _CliError("output must differ from input")
    regions = tuple(cast(list[Region], getattr(args, "crop", [])))
    region_ids = [region.region_id for region in regions]
    duplicates = sorted({region_id for region_id in region_ids if region_ids.count(region_id) > 1})
    if duplicates:
        raise _CliError(f"duplicate crop ID: {duplicates[0]}")
    return RunConfig(
        input_path=input_path,
        output_dir=output_dir,
        regions=regions,
        threshold=cast(float, getattr(args, "threshold", 0.3)),
        max_frames=cast(int | None, getattr(args, "max_frames", None)),
        warmup_runs=cast(int, getattr(args, "warmup_runs", 2)),
        annotate_every=cast(int, getattr(args, "annotate_every", 10)),
        device=cast(str, getattr(args, "device", "auto")),
    )


def _apply_config_overrides(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        input_path=config.input_path,
        output_dir=cast(Path, args.output) if hasattr(args, "output") else config.output_dir,
        regions=(
            tuple(cast(list[Region], args.crop)) if hasattr(args, "crop") else config.regions
        ),
        threshold=(
            cast(float, args.threshold) if hasattr(args, "threshold") else config.threshold
        ),
        max_frames=(
            cast(int | None, args.max_frames) if hasattr(args, "max_frames") else config.max_frames
        ),
        warmup_runs=(
            cast(int, args.warmup_runs) if hasattr(args, "warmup_runs") else config.warmup_runs
        ),
        annotate_every=(
            cast(int, args.annotate_every)
            if hasattr(args, "annotate_every")
            else config.annotate_every
        ),
        detector_id=config.detector_id,
        device=cast(str, args.device) if hasattr(args, "device") else config.device,
        capture=config.capture,
    )


def _compare_command(args: argparse.Namespace) -> int:
    report = compare_runs(
        cast(Path, args.left),
        cast(Path, args.right),
        box_atol=cast(float, args.box_atol),
        score_atol=cast(float, args.score_atol),
    )
    equivalent = report.get("equivalent") is True
    print(
        f"equivalent={str(equivalent).lower()} "
        f"left={report.get('left_detection_count')} "
        f"right={report.get('right_detection_count')} "
        f"mismatches={report.get('mismatch_count')}"
    )
    return 0 if equivalent else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Parse and execute one CLI command, returning a process exit code."""

    try:
        args = _build_parser().parse_args(argv)
        if args.command == "run":
            return _run_command(args)
        if args.command == "compare":
            return _compare_command(args)
        raise _CliError("a command is required")
    except (_CliError, OSError, RuntimeError, ValueError) as error:
        message = " ".join(str(error).splitlines()) or type(error).__name__
        print(f"error: {message}", file=sys.stderr)
        return 2
