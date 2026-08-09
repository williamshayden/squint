from __future__ import annotations

import argparse
from collections.abc import Sequence

from squint_rl import __version__


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="squint")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)
    if args.version:
        print(f"squint {__version__}")
    return 0
