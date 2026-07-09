"""Inference and frame-level scoring pipeline for Z flow DiT."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.pipelines.training_pipeline import (
    configure_torch_backend,
    get_device,
    load_checkpoint_for_inference,
    maybe_compile_model,
)
from src.pipelines.z_dit_pipeline import (
    FeatureZDataset,
    aggregate_frame_scores,
    apply_score_calibration,
    apply_score_normalization,
    apply_track_grid_ablation,
    apply_track_grid_channel_mask,
    apply_track_grid_gate_override,
    build_flow_matching_sampler,
    compute_binary_metrics,
    compute_score_metrics,
    condition_uses_appearance_features,
    condition_uses_legacy_track_gate,
    condition_uses_track_features,
    config_to_dict,
    display_track_ablation_mode,
    fit_score_calibration,
    load_feature_records,
    load_flow_matching_config,
    load_json,
    load_pipeline_config,
    make_loader_kwargs,
    move_feature_batch,
    move_track_context_grid,
    per_frame_z_score_variants,
    save_jsonl,
    write_json,
    write_yaml,
)
from src.utils import cleanup_memory, progress_bar


def infer(
    config_path: Path,
    *,
    checkpoint_path: Path | None = None,
    run_id: str | None = None,
    output_run_id: str | None = None,
    feature_index: Path | None = None,
    calibration_stats_path: Path | None = None,
    fit_calibration_path: Path | None = None,
    limit_samples: int | None = None,
    overwrite: bool | None = None,
) -> Path:
    """Run Z DiT inference and write 04_predictions artifacts."""

    config = load_pipeline_config(config_path)
    active_checkpoint = resolve_checkpoint_path(config, checkpoint_path, run_id)
    active_run_id = output_run_id or run_id or active_checkpoint.parent.name
    output_dir = config.inference.output_root / active_run_id
    effective_overwrite = config.inference.overwrite if overwrite is None else overwrite
    if (
        output_dir.exists()
        and not effective_overwrite
        and (output_dir / "frame_scores.jsonl").is_file()
    ):
        raise FileExistsError(f"Prediction output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    requires_track_features = condition_uses_track_features(config.model.condition)
    requires_low_features = condition_uses_appearance_features(config.model.condition)
    records = load_feature_records(
        feature_index or config.inference.feature_index,
        normal_only=False,
        limit_samples=limit_samples,
        require_track_features=requires_track_features,
        require_low_features=requires_low_features,
    )
    dataset = FeatureZDataset(
        records,
        require_track_features=requires_track_features,
        require_low_features=requires_low_features,
    )
    loader = DataLoader(
        dataset,
        **make_loader_kwargs(
            config.inference.batch_size,
            config.inference.num_workers,
            shuffle=False,
            prefetch_factor=config.optimization.prefetch_factor,
        ),
    )

    configure_torch_backend(config.optimization)
    device = get_device()
    model, z_stats, checkpoint_config = load_checkpoint_for_inference(
        active_checkpoint,
        device=device,
    )
    track_grid_gate_value = None
    if condition_uses_legacy_track_gate(config.model.condition):
        track_grid_gate_value = apply_track_grid_gate_override(
            model,
            config.model.condition.track_grid.gate_override,
        )
    model = maybe_compile_model(model, enabled=config.inference.compile)
    checkpoint_flow_config = load_flow_matching_config(checkpoint_config["flow_matching"])
    flow_config = replace(
        checkpoint_flow_config,
        inference_steps=config.flow_matching.inference_steps,
    )
    flow = build_flow_matching_sampler(flow_config)

    prediction_records = []
    tensor_dir = output_dir / "tensors" / "z_hat"
    progress = progress_bar(loader, desc="infer Z DiT", unit="batch")
    try:
        for batch in progress:
            high, z, low = move_feature_batch(batch, device)
            context_track_grid = move_track_context_grid(
                batch,
                device,
                enabled=condition_uses_track_features(config.model.condition),
                z_patch_size=config.model.z_adapter.patch_size,
            )
            if context_track_grid is not None:
                context_track_grid = apply_track_grid_ablation(
                    context_track_grid,
                    config.model.condition.track_grid.ablation_mode,
                    config.model.condition.track_grid.shuffle_seed,
                )
                context_track_grid = apply_track_grid_channel_mask(
                    context_track_grid,
                    config.model.condition.track_grid.channel_mask,
                )
            track = batch.get("track")
            future_track_grid = (
                track["future_grid"].to(device, non_blocking=True) if track is not None else None
            )
            with torch.inference_mode():
                z_hat = flow.euler_sample(
                    model,
                    high,
                    tuple(z.shape),
                    low_features=low,
                    track_features=context_track_grid,
                    z_stats=z_stats,
                    inference_steps=config.flow_matching.inference_steps,
                )
                score_variants = per_frame_z_score_variants(
                    z,
                    z_hat,
                    scoring=config.scoring,
                    patch_size=config.model.z_adapter.patch_size,
                    z_stats=z_stats,
                    low=low,
                    track_grid=future_track_grid,
                )
            batch_records = build_prediction_records(
                batch["metadata"],
                {key: value.detach().cpu() for key, value in score_variants.items()},
                primary_variant=config.scoring.variant,
                z_hat=z_hat.detach().cpu(),
                tensor_dir=tensor_dir,
                save_tensors=config.inference.save_tensors,
            )
            prediction_records.extend(batch_records)
            progress.set_postfix(samples=len(prediction_records), refresh=False)
    finally:
        progress.close()
        cleanup_memory(cuda=device.type == "cuda")

    frame_scores = apply_score_normalization(
        aggregate_frame_scores(prediction_records),
        scoring=config.scoring,
    )
    active_fit_calibration_path = fit_calibration_path or config.scoring.calibration.fit_output_path
    if active_fit_calibration_path is not None:
        write_json(
            active_fit_calibration_path,
            fit_score_calibration(frame_scores, scoring=config.scoring),
        )
    active_calibration_stats_path = calibration_stats_path or config.scoring.calibration.stats_path
    if active_calibration_stats_path is not None:
        frame_scores = apply_score_calibration(
            frame_scores,
            scoring=config.scoring,
            calibration_stats=load_json(active_calibration_stats_path),
        )
    elif config.scoring.score_normalization.primary == "scene_calibrated":
        raise ValueError(
            "score_normalization.primary=scene_calibrated requires calibration stats. "
            "Pass --calibration-stats or set scoring.calibration.stats_path."
        )
    metrics = compute_binary_metrics(frame_scores)
    metrics.update(
        {
            "run_id": active_run_id,
            "checkpoint_path": str(active_checkpoint),
            "prediction_records": len(prediction_records),
            "primary_score": config.scoring.score_normalization.primary,
            "track_grid_ablation": track_grid_ablation_metadata(
                config,
                gate_value=track_grid_gate_value,
            ),
            "score_metrics": compute_score_metrics(frame_scores),
        }
    )

    write_yaml(output_dir / "config_resolved.yaml", config_to_dict(config))
    write_json(output_dir / "checkpoint_config.json", checkpoint_config)
    (output_dir / "checkpoint_ref.txt").write_text(str(active_checkpoint) + "\n", encoding="utf-8")
    save_jsonl(output_dir / "future_frame_predictions.jsonl", prediction_records)
    save_jsonl(output_dir / "frame_scores.jsonl", frame_scores)
    write_json(output_dir / "metrics.json", metrics)
    print(f"wrote predictions -> {output_dir}", flush=True)
    return output_dir


def track_grid_ablation_metadata(config: Any, *, gate_value: float | None) -> dict[str, Any]:
    """Return inference-time C_track ablation metadata."""

    track_grid = config.model.condition.track_grid
    return {
        "enabled": condition_uses_track_features(config.model.condition),
        "legacy_track_gate": condition_uses_legacy_track_gate(config.model.condition),
        "condition_mode": config.model.condition.mode,
        "ablation_mode": display_track_ablation_mode(track_grid.ablation_mode),
        "channel_mask": track_grid.channel_mask,
        "shuffle_seed": int(track_grid.shuffle_seed),
        "gate_override": track_grid.gate_override,
        "gate_value": gate_value,
    }


def resolve_checkpoint_path(
    config: Any,
    checkpoint_path: Path | None,
    run_id: str | None,
) -> Path:
    """Resolve the checkpoint path for inference."""

    if checkpoint_path is not None:
        return checkpoint_path
    if config.inference.checkpoint_path is not None:
        return config.inference.checkpoint_path
    if run_id:
        return config.training.output_root / run_id / "best.pt"
    raise ValueError("Provide --checkpoint, inference.checkpoint_path, or --run-id")


def build_prediction_records(
    metadata: list[dict[str, Any]],
    score_variants: dict[str, torch.Tensor],
    primary_variant: str,
    z_hat: torch.Tensor,
    *,
    tensor_dir: Path,
    save_tensors: bool,
) -> list[dict[str, Any]]:
    """Build sample-level prediction records with future frame scores."""

    records = []
    for index, sample in enumerate(metadata):
        variant_scores = {
            variant: [float(value) for value in scores[index].tolist()]
            for variant, scores in score_variants.items()
        }
        frame_scores = variant_scores[primary_variant]
        record = {
            "sample_id": sample["sample_id"],
            "video_id": sample["video_id"],
            "scene_id": sample.get("scene_id"),
            "future_frames": [int(frame_idx) for frame_idx in sample["future_frames"]],
            "future_frame_labels": [int(label) for label in sample["future_frame_labels"]],
            "future_frame_scores": frame_scores,
            "future_frame_score_variants": variant_scores,
            "sample_score": sum(frame_scores) / len(frame_scores),
            "sample_score_variants": {
                variant: sum(values) / len(values) for variant, values in variant_scores.items()
            },
        }
        if save_tensors:
            path = tensor_dir / f"{sample['sample_id']}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "sample_id": sample["sample_id"],
                    "z_hat": z_hat[index].to(torch.float16),
                },
                path,
            )
            record["z_hat_path"] = str(path)
        records.append(record)
    return records
