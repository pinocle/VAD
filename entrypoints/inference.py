"""Run Z DiT inference and write frame-level anomaly scores."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.inference_pipeline import infer  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/local.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-run-id", type=str, default=None)
    parser.add_argument("--feature-index", type=Path, default=None)
    parser.add_argument("--calibration-stats", type=Path, default=None)
    parser.add_argument("--fit-calibration", type=Path, default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    output_dir = infer(
        args.config,
        checkpoint_path=args.checkpoint,
        run_id=args.run_id,
        output_run_id=args.output_run_id,
        feature_index=args.feature_index,
        calibration_stats_path=args.calibration_stats,
        fit_calibration_path=args.fit_calibration,
        limit_samples=args.limit_samples,
        overwrite=args.overwrite,
    )
    print(f"inference complete -> {output_dir}")


if __name__ == "__main__":
    main()
