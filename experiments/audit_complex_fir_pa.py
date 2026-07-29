"""CLI entry point for the causal proper-complex FIR residual PA audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.audit_widely_linear_pa import run_from_config


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered proper-complex FIR PA residual audit "
            "without test access."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    run_from_config(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
