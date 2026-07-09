"""Track-grid condition ablation experiment runner."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.pipelines.inference_pipeline import infer, resolve_checkpoint_path
from src.pipelines.z_dit_pipeline import (
    SCORE_VARIANT_GLOBAL,
    SCORE_VARIANT_GLOBAL_PATCH,
    SCORE_VARIANT_LOW_WEIGHTED,
    SCORE_VARIANT_MEAN_TOPK_PLUS_TRACK_REGION,
    SCORE_VARIANT_MOTION_GLOBAL_PATCH,
    SCORE_VARIANT_MOTION_PATCH,
    SCORE_VARIANT_PATCH,
    SCORE_VARIANT_TRACK_REGION_TOPK,
    SCORE_VARIANT_TRACK_WEIGHTED,
    display_track_ablation_mode,
    load_pipeline_config,
    normalize_track_ablation_mode,
    write_json,
)

ALL_SCORING_VARIANTS = [
    SCORE_VARIANT_GLOBAL,
    SCORE_VARIANT_PATCH,
    SCORE_VARIANT_LOW_WEIGHTED,
    SCORE_VARIANT_GLOBAL_PATCH,
    SCORE_VARIANT_MOTION_PATCH,
    SCORE_VARIANT_MOTION_GLOBAL_PATCH,
    SCORE_VARIANT_TRACK_WEIGHTED,
    SCORE_VARIANT_TRACK_REGION_TOPK,
    SCORE_VARIANT_MEAN_TOPK_PLUS_TRACK_REGION,
]

SUMMARY_COLUMNS = [
    "run_id",
    "output_dir",
    "checkpoint",
    "ablation_mode",
    "channel_mask",
    "gate_value",
    "global_patch_video_centered_auc",
    "global_patch_video_centered_ap",
    "track_weighted_video_centered_auc",
    "track_weighted_video_centered_ap",
    "global_patch_rolling_centered_auc",
    "global_patch_rolling_centered_ap",
    "track_weighted_rolling_centered_auc",
    "track_weighted_rolling_centered_ap",
    "primary_score_auc",
    "primary_score_ap",
]


@dataclass(frozen=True)
class AblationSpec:
    """One track-grid ablation inference configuration."""

    ablation_mode: str
    channel_mask: str
    gate_override: float | None

    @property
    def display_mode(self) -> str:
        return display_track_ablation_mode(self.ablation_mode)

    @property
    def gate_label(self) -> str:
        return "ckpt" if self.gate_override is None else format_float_token(self.gate_override)


def run_trackgrid_ablation_experiment(
    *,
    config_path: Path,
    checkpoint_path: Path | None,
    run_id: str | None,
    feature_index: Path | None,
    output_prefix: str,
    ablation_modes: list[str],
    channel_masks: list[str],
    gate_values: list[float | None],
    shuffle_seed: int,
    summary_dir: Path | None,
    limit_samples: int | None,
    overwrite: bool,
) -> list[dict[str, Any]]:
    """Run track-grid ablations and write compact summary artifacts."""

    base_config = load_pipeline_config(config_path)
    active_checkpoint = resolve_checkpoint_path(base_config, checkpoint_path, run_id)
    active_summary_dir = (
        summary_dir or base_config.inference.output_root / f"{output_prefix}_summary"
    )
    active_summary_dir.mkdir(parents=True, exist_ok=True)
    config_dir = active_summary_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    raw_config = load_yaml_dict(config_path)
    rows = []
    for spec in build_ablation_specs(
        ablation_modes=ablation_modes,
        channel_masks=channel_masks,
        gate_values=gate_values,
    ):
        output_run_id = make_output_run_id(output_prefix, spec)
        variant_config = prepare_ablation_config(
            raw_config,
            spec=spec,
            feature_index=feature_index,
            shuffle_seed=shuffle_seed,
        )
        variant_config_path = config_dir / f"{output_run_id}.yaml"
        write_yaml(variant_config_path, variant_config)
        output_dir = infer(
            variant_config_path,
            checkpoint_path=active_checkpoint,
            output_run_id=output_run_id,
            feature_index=feature_index,
            limit_samples=limit_samples,
            overwrite=overwrite,
        )
        row = collect_summary_row(
            output_dir,
            run_id=output_run_id,
            checkpoint_path=active_checkpoint,
            spec=spec,
        )
        rows.append(row)

    write_summary(active_summary_dir, rows)
    print(f"wrote track-grid ablation summary -> {active_summary_dir}")
    return rows


def build_ablation_specs(
    *,
    ablation_modes: list[str],
    channel_masks: list[str],
    gate_values: list[float | None],
) -> list[AblationSpec]:
    """Build the cartesian product of requested ablation settings."""

    specs = []
    for mode in ablation_modes:
        normalized_mode = normalize_track_ablation_mode(mode)
        for channel_mask in channel_masks:
            for gate_value in gate_values:
                specs.append(
                    AblationSpec(
                        ablation_mode=normalized_mode,
                        channel_mask=channel_mask,
                        gate_override=gate_value,
                    )
                )
    return specs


def prepare_ablation_config(
    raw_config: dict[str, Any],
    *,
    spec: AblationSpec,
    feature_index: Path | None,
    shuffle_seed: int,
) -> dict[str, Any]:
    """Return a config dict for one ablation inference run."""

    config = deep_copy_jsonable(raw_config)
    model = config.setdefault("model", {})
    condition = model.setdefault("condition", {})
    condition["use_track_grid"] = True
    condition.setdefault("track_grid_gate", 0.1)
    track_grid = condition.setdefault("track_grid", {})
    track_grid["ablation_mode"] = spec.ablation_mode
    track_grid["shuffle_seed"] = int(shuffle_seed)
    track_grid["channel_mask"] = spec.channel_mask
    track_grid["gate_override"] = spec.gate_override

    inference = config.setdefault("inference", {})
    if feature_index is not None:
        inference["feature_index"] = str(feature_index)

    scoring = config.setdefault("scoring", {})
    scoring["variant"] = SCORE_VARIANT_TRACK_WEIGHTED
    current_variants = [str(value) for value in scoring.get("variants", [])]
    merged = []
    for variant in current_variants + ALL_SCORING_VARIANTS:
        if variant not in merged:
            merged.append(variant)
    scoring["variants"] = merged
    score_normalization = scoring.setdefault("score_normalization", {})
    score_normalization.setdefault("primary", "video_centered")
    return config


def collect_summary_row(
    output_dir: Path,
    *,
    run_id: str,
    checkpoint_path: Path,
    spec: AblationSpec,
) -> dict[str, Any]:
    """Collect one compact metrics row from an inference output directory."""

    metrics = load_json(output_dir / "metrics.json")
    score_metrics = metrics.get("score_metrics", {})
    ablation_metadata = metrics.get("track_grid_ablation", {})
    row: dict[str, Any] = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "ablation_mode": spec.display_mode,
        "channel_mask": spec.channel_mask,
        "gate_value": ablation_metadata.get("gate_value"),
    }
    add_metric_pair(
        row,
        score_metrics,
        source_key="global_patch_video_centered_score",
        prefix="global_patch_video_centered",
    )
    add_metric_pair(
        row,
        score_metrics,
        source_key="track_weighted_video_centered_score",
        prefix="track_weighted_video_centered",
    )
    add_metric_pair(
        row,
        score_metrics,
        source_key="global_patch_rolling_centered_score",
        prefix="global_patch_rolling_centered",
    )
    add_metric_pair(
        row,
        score_metrics,
        source_key="track_weighted_rolling_centered_score",
        prefix="track_weighted_rolling_centered",
    )
    add_metric_pair(row, score_metrics, source_key="score", prefix="primary_score")
    return row


def add_metric_pair(
    row: dict[str, Any],
    score_metrics: dict[str, Any],
    *,
    source_key: str,
    prefix: str,
) -> None:
    """Add AUC/AP metrics for one score key to a summary row."""

    metric = score_metrics.get(source_key, {})
    row[f"{prefix}_auc"] = metric.get("roc_auc")
    row[f"{prefix}_ap"] = metric.get("average_precision")


def write_summary(summary_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV and JSON ablation summaries."""

    summary_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary_dir / "summary.json", {"runs": rows})
    with (summary_dir / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_output_run_id(output_prefix: str, spec: AblationSpec) -> str:
    """Return a stable output run id for one ablation spec."""

    parts = [
        output_prefix,
        spec.display_mode,
        sanitize_token(spec.channel_mask),
        f"gate{spec.gate_label}",
    ]
    return "_".join(part for part in parts if part)


def sanitize_token(value: str) -> str:
    """Return a filesystem/run-id friendly token."""

    return "".join(char if char.isalnum() else "_" for char in str(value)).strip("_")


def format_float_token(value: float) -> str:
    """Format a float for stable run ids."""

    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def parse_csv_list(value: str) -> list[str]:
    """Parse a comma-separated CLI list."""

    return [item.strip() for item in value.split(",") if item.strip()]


def parse_gate_values(value: str | None) -> list[float | None]:
    """Parse optional gate overrides from CLI."""

    if value is None or not value.strip():
        return [None]
    values: list[float | None] = []
    for item in parse_csv_list(value):
        if item.lower() in {"none", "null", "ckpt", "checkpoint"}:
            values.append(None)
        else:
            values.append(float(item))
    return values


def deep_copy_jsonable(value: dict[str, Any]) -> dict[str, Any]:
    """Copy a YAML/JSON-compatible dictionary."""

    return json.loads(json.dumps(value))


def load_yaml_dict(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""

    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""

    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write YAML payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description="Run track-grid ablation inference sweeps.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--feature-index", type=Path, default=None)
    parser.add_argument("--output-prefix", type=str, default="trackgrid_ablation")
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument(
        "--ablation-modes",
        type=str,
        default="real_track,zero_track,shuffled_track",
        help="Comma-separated modes: real_track, zero_track, shuffled_track.",
    )
    parser.add_argument(
        "--channel-masks",
        type=str,
        default="all_channels",
        help="Comma-separated masks such as all_channels,objectness_only,speed_only.",
    )
    parser.add_argument(
        "--gate-values",
        type=str,
        default="",
        help="Comma-separated gate overrides. Empty or 'ckpt' uses checkpoint gate.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""

    args = parse_args(argv)
    run_trackgrid_ablation_experiment(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        run_id=args.run_id,
        feature_index=args.feature_index,
        output_prefix=args.output_prefix,
        ablation_modes=parse_csv_list(args.ablation_modes),
        channel_masks=parse_csv_list(args.channel_masks),
        gate_values=parse_gate_values(args.gate_values),
        shuffle_seed=args.shuffle_seed,
        summary_dir=args.summary_dir,
        limit_samples=args.limit_samples,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
