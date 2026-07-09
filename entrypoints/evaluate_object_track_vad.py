"""Evaluate an object/track-only VAD baseline from detector/tracker caches."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.object_track_vad_pipeline import (  # noqa: E402
    FRAME_AGGREGATIONS,
    PROTOTYPE_METHODS,
    SCORE_MODES,
    evaluate_object_track_vad,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-track-cache", type=Path, required=True)
    parser.add_argument("--test-track-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--test-feature-index", type=Path, default=None)
    parser.add_argument("--train-labels", type=Path, default=None)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--max-prototypes", type=int, default=2048)
    parser.add_argument("--prototype-method", choices=PROTOTYPE_METHODS, default="random")
    parser.add_argument("--primary-score-mode", choices=SCORE_MODES, default="class_scene_memory_distance")
    parser.add_argument("--frame-aggregation", choices=FRAME_AGGREGATIONS, default="topk_mean")
    parser.add_argument("--frame-top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    output_dir = evaluate_object_track_vad(
        train_track_cache=args.train_track_cache,
        test_track_cache=args.test_track_cache,
        output_dir=args.output_dir,
        labels_path=args.labels,
        test_feature_index=args.test_feature_index,
        train_labels_path=args.train_labels,
        context_length=args.context_length,
        max_prototypes=args.max_prototypes,
        prototype_method=args.prototype_method,
        primary_score_mode=args.primary_score_mode,
        frame_aggregation=args.frame_aggregation,
        frame_top_k=args.frame_top_k,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"object-track VAD evaluation complete -> {output_dir}")


if __name__ == "__main__":
    main()
