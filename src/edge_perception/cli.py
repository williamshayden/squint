"""Standard-library command line interface for checkpoint runs and comparison."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from math import isfinite
from pathlib import Path
from typing import NoReturn, cast

from edge_perception.compare import compare_runs
from edge_perception.contracts import Region
from edge_perception.runner import RunConfig, run_checkpoint, validate_output_directory


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
    run.add_argument("input", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--crop", action="append", type=_crop, default=[])
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument("--threshold", type=_threshold, default=0.3)
    run.add_argument("--warmup-runs", type=_nonnegative("warmup-runs"), default=2)
    run.add_argument("--max-frames", type=_nonnegative("max-frames"))
    run.add_argument("--annotate-every", type=_nonnegative("annotate-every"), default=10)

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
    input_path = cast(Path, args.input).resolve()
    output_dir = cast(Path, args.output).resolve()
    regions = tuple(cast(list[Region], args.crop))
    if not input_path.is_file():
        raise _CliError(f"input video does not exist: {input_path}")
    if output_dir == input_path:
        raise _CliError("output must differ from input")
    validate_output_directory(output_dir)
    region_ids = [region.region_id for region in regions]
    duplicates = sorted({region_id for region_id in region_ids if region_ids.count(region_id) > 1})
    if duplicates:
        raise _CliError(f"duplicate crop ID: {duplicates[0]}")

    threshold = cast(float, args.threshold)
    config = RunConfig(
        input_path=input_path,
        output_dir=output_dir,
        regions=regions,
        threshold=threshold,
        max_frames=cast(int | None, args.max_frames),
        warmup_runs=cast(int, args.warmup_runs),
        annotate_every=cast(int, args.annotate_every),
    )

    from edge_perception.detectors.dfine import DfineDetector

    detector = DfineDetector.load(
        threshold=threshold,
        device=cast(str, args.device),
    )
    summary = run_checkpoint(config, detector)
    print(f"output: {output_dir}")
    print(
        f"status={summary.get('status')} "
        f"frames={summary.get('frames_processed')} "
        f"inferences={summary.get('inference_count')}"
    )
    return 0


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
