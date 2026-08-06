"""Internal JSONL worker process for configured checkpoint runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter_ns
from typing import NoReturn, TextIO

from edge_perception.config import load_run_config
from edge_perception.detector import Detector
from edge_perception.detectors.registry import load_detector
from edge_perception.progress import ProgressEvent
from edge_perception.runner import run_checkpoint


class _WorkerFailure(Exception):
    def __init__(self, error: Exception, *, runner_started: bool) -> None:
        super().__init__(str(error))
        self.error = error
        self.runner_started = runner_started


class _WorkerArgumentError(ValueError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _WorkerArgumentError(message)


def _emit_json(stream: TextIO, event: ProgressEvent) -> None:
    stream.write(json.dumps(event.to_dict(), allow_nan=False, sort_keys=True) + "\n")
    stream.flush()


def run_worker(
    config_path: Path,
    cancel_file: Path,
    *,
    detector_loader: Callable[..., Detector] = load_detector,
    stream: TextIO = sys.stdout,
) -> int:
    """Load a configured detector and run it with JSONL progress output."""

    runner_started = False
    try:
        config = load_run_config(config_path)

        def emit(event: ProgressEvent) -> None:
            _emit_json(stream, event)

        detector = detector_loader(
            config.detector_id,
            threshold=config.threshold,
            device=config.device,
        )
        runner_started = True
        summary = run_checkpoint(
            config,
            detector,
            progress=emit,
            cancel_requested=cancel_file.exists,
        )
        return 0 if summary["status"] in {"complete", "cancelled"} else 2
    except Exception as error:
        raise _WorkerFailure(error, runner_started=runner_started) from error


def _build_parser() -> _Parser:
    parser = _Parser(prog="python -m edge_perception.worker", add_help=False)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cancel-file", required=True, type=Path)
    return parser


def _compact_error(error: Exception) -> str:
    return " ".join(str(error).splitlines()) or type(error).__name__


def _report_failure(
    error: Exception,
    *,
    runner_started: bool,
    started_ns: int,
) -> int:
    message = _compact_error(error)
    if not runner_started:
        _emit_json(
            sys.stdout,
            ProgressEvent(
                phase="failed",
                frames_processed=0,
                inference_count=0,
                elapsed_ms=(perf_counter_ns() - started_ns) / 1_000_000.0,
                error=f"{type(error).__name__}: {message}",
            ),
        )
    print(f"error: {message}", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the internal worker command and return its process exit code."""

    started_ns = perf_counter_ns()
    try:
        args = _build_parser().parse_args(argv)
        return run_worker(args.config, args.cancel_file, stream=sys.stdout)
    except _WorkerArgumentError as error:
        return _report_failure(error, runner_started=False, started_ns=started_ns)
    except _WorkerFailure as failure:
        return _report_failure(
            failure.error,
            runner_started=failure.runner_started,
            started_ns=started_ns,
        )


if __name__ == "__main__":
    raise SystemExit(main())
