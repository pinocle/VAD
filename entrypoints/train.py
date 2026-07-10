"""Train the memory-guided fixed-grid RGB-patch DiT."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.training_pipeline import train  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/local.yaml"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    run_dir = train(
        args.config,
        run_id=args.run_id,
        max_steps=args.max_steps,
        limit_samples=args.limit_samples,
        overwrite=args.overwrite,
    )
    print(f"training complete -> {run_dir}")


if __name__ == "__main__":
    main()
