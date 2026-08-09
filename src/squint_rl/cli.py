from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from squint_rl import __version__
from squint_rl.benchmark import evaluate
from squint_rl.config import BenchmarkConfig, ConfigurationError
from squint_rl.episode import Episode, EpisodeValidationError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="squint")
    parser.add_argument("--version", action="store_true")
    commands = parser.add_subparsers(dest="command")
    episode = commands.add_parser("episode")
    episode_commands = episode.add_subparsers(dest="episode_command")
    validate = episode_commands.add_parser("validate")
    validate.add_argument("episode")
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("config")
    benchmark.add_argument("--policy")
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else 2
    if args.version:
        print(f"squint {__version__}")
        return 0
    try:
        if args.command == "episode" and args.episode_command == "validate":
            Episode.open(args.episode)
            print(
                json.dumps(
                    {"status": "valid", "episode": str(Path(args.episode).resolve())},
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "benchmark":
            config = BenchmarkConfig.load(args.config)
            if args.policy is not None:
                config = config.with_policy_override(args.policy)
            result = evaluate(config)
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "output_dir": str(result.output_dir),
                        "action_sha256": result.action_sha256,
                        "metric_input_sha256": result.metric_input_sha256,
                    },
                    sort_keys=True,
                )
            )
            return 0
        parser.print_usage()
        return 2
    except (ConfigurationError, EpisodeValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 - evaluator backends surface heterogeneous runtime errors.
        print(f"error: {error}", file=sys.stderr)
        return 1
