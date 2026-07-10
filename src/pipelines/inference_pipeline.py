"""Noise-to-RGB-patch inference and frame-level anomaly scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from src.models import FlowMatcher
from src.pipelines.rgb_patch_pipeline import (
    RGBPatchDataset,
    aggregate_frame_scores,
    compute_binary_metrics,
    config_to_dict,
    infer_patch_shape,
    load_pipeline_config,
    load_sample_records,
    make_loader,
    score_patch_prediction,
    unpatchify_rgb,
    write_jsonl,
)
from src.pipelines.training_pipeline import (
    get_device,
    load_checkpoint_for_inference,
    move_batch,
)
from src.utils import cleanup_memory, progress_bar


def infer(
    config_path: Path,
    *,
    checkpoint_path: Path | None = None,
    run_id: str | None = None,
    output_run_id: str | None = None,
    sample_index: Path | None = None,
    limit_samples: int | None = None,
    overwrite: bool | None = None,
) -> Path:
    """Generate future RGB patches from noise and write frame-level anomaly scores."""

    config = load_pipeline_config(config_path)
    active_checkpoint = resolve_checkpoint_path(config, checkpoint_path, run_id)
    active_run_id = output_run_id or run_id or active_checkpoint.parent.name
    output_dir = config.inference.output_root / active_run_id
    active_overwrite = config.inference.overwrite if overwrite is None else overwrite
    if output_dir.exists() and any(output_dir.iterdir()) and not active_overwrite:
        raise FileExistsError(f"Prediction output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_sample_records(
        sample_index or config.inference.sample_index,
        normal_only=False,
        limit_samples=limit_samples,
    )
    dataset = RGBPatchDataset(records, config.model)
    expected_shape = infer_patch_shape(dataset[0], config.model)
    loader = make_loader(
        dataset,
        batch_size=config.inference.batch_size,
        num_workers=config.inference.num_workers,
        shuffle=False,
    )
    device = get_device()
    model, checkpoint_shape, checkpoint_model_config, checkpoint_config = (
        load_checkpoint_for_inference(
            active_checkpoint,
            device=device,
        )
    )
    validate_checkpoint_contract(
        config.model, expected_shape, checkpoint_model_config, checkpoint_shape
    )
    if config.inference.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    flow = FlowMatcher(inference_steps=config.flow_matching.inference_steps)

    prediction_records = []
    tensor_root = output_dir / "tensors"
    progress = progress_bar(loader, desc="infer RGB patch DiT", unit="batch")
    try:
        for batch in progress:
            context, target = move_batch(batch, device)
            with torch.inference_mode():
                generated, memory_distance = flow.sample(
                    model,
                    context=context,
                    target_shape=tuple(target.shape),
                    inference_steps=config.flow_matching.inference_steps,
                )
                score_components = score_patch_prediction(
                    target,
                    generated,
                    scoring=config.scoring,
                    memory_distance=memory_distance,
                )
            batch_records = build_prediction_records(
                batch["metadata"],
                generated.detach().cpu(),
                score_components,
                tensor_root=tensor_root,
                save_tensors=config.inference.save_tensors,
                image_size=checkpoint_shape.image_size,
                patch_size=checkpoint_shape.patch_size,
            )
            prediction_records.extend(batch_records)
            progress.set_postfix(samples=len(prediction_records), refresh=False)
    finally:
        progress.close()
        cleanup_memory(cuda=device.type == "cuda")

    frame_scores = aggregate_frame_scores(prediction_records)
    metrics = compute_binary_metrics(frame_scores)
    metrics.update(
        {
            "run_id": active_run_id,
            "checkpoint_path": str(active_checkpoint),
            "prediction_records": len(prediction_records),
        }
    )
    write_yaml(output_dir / "config_resolved.yaml", config_to_dict(config))
    write_json(output_dir / "checkpoint_config.json", checkpoint_config)
    write_jsonl(output_dir / "future_frame_predictions.jsonl", prediction_records)
    write_jsonl(output_dir / "frame_scores.jsonl", frame_scores)
    write_json(output_dir / "metrics.json", metrics)
    return output_dir


def resolve_checkpoint_path(
    config: Any,
    checkpoint_path: Path | None,
    run_id: str | None,
) -> Path:
    """Resolve an explicit checkpoint, config checkpoint, or best run checkpoint."""

    if checkpoint_path is not None:
        return checkpoint_path
    if config.inference.checkpoint_path is not None:
        return config.inference.checkpoint_path
    if run_id:
        return config.training.output_root / run_id / "best.pt"
    raise ValueError("Provide --checkpoint, inference.checkpoint_path, or --run-id")


def validate_checkpoint_contract(
    config_model: Any,
    expected_shape: Any,
    checkpoint_model: Any,
    checkpoint_shape: Any,
) -> None:
    """Reject incompatible image/patch grids before expensive inference."""

    if config_model.image_size != checkpoint_model.image_size:
        raise ValueError("checkpoint model.image_size does not match inference config")
    if config_model.patch_size != checkpoint_model.patch_size:
        raise ValueError("checkpoint model.patch_size does not match inference config")
    if expected_shape != checkpoint_shape:
        raise ValueError(
            "processed sample windows do not match checkpoint RGB patch shape: "
            f"samples={expected_shape}, checkpoint={checkpoint_shape}"
        )


def build_prediction_records(
    metadata: list[dict[str, Any]],
    generated: torch.Tensor,
    score_components: dict[str, torch.Tensor],
    *,
    tensor_root: Path,
    save_tensors: bool,
    image_size: int,
    patch_size: int,
) -> list[dict[str, Any]]:
    """Build JSON-serializable sample predictions and optionally store RGB frames."""

    records = []
    for index, sample in enumerate(metadata):
        record = {
            "sample_id": sample["sample_id"],
            "video_id": sample["video_id"],
            "future_frames": [int(value) for value in sample["future_frames"]],
            "future_frame_labels": [int(value) for value in sample["future_frame_labels"]],
            "scores": [float(value) for value in score_components["score"][index].cpu().tolist()],
            "full_frame_scores": [
                float(value) for value in score_components["full_frame"][index].cpu().tolist()
            ],
            "topk_scores": [
                float(value) for value in score_components["topk"][index].cpu().tolist()
            ],
            "memory_distances": [
                float(value) for value in score_components["memory_distance"][index].cpu().tolist()
            ],
        }
        if save_tensors:
            tensor_path = tensor_root / f"{sample['sample_id']}.pt"
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "sample_id": sample["sample_id"],
                    "predicted_patches": generated[index].to(torch.float16),
                    "predicted_rgb": unpatchify_rgb(
                        generated[index],
                        image_size=image_size,
                        patch_size=patch_size,
                    ).to(torch.float16),
                },
                tensor_path,
            )
            record["prediction_path"] = str(tensor_path)
        records.append(record)
    return records


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write UTF-8 JSON artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    """Write resolved configuration as YAML."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
