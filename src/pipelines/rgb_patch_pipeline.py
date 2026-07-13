"""Fixed-grid RGB patch data, configuration, and scoring utilities for VAD."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.utils import configure_progress_from_config

RGB_CHANNELS = 3
SUPPORTED_PATCH_SIZES = {16, 32}
MODEL_NAME = "rgb_patch_dit"


@dataclass(frozen=True)
class RGBPatchModelConfig:
    """Architecture and fixed-grid patch settings."""

    image_size: int
    patch_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    memory_size: int
    memory_temperature: float
    # Continuous flow time stays in [0, 1]; only its sinusoidal representation is scaled.
    time_embedding_scale: float = 1.0


@dataclass(frozen=True)
class FlowMatchingConfig:
    """Continuous flow-matching sampling settings."""

    inference_steps: int
    timestep_distribution: str


@dataclass(frozen=True)
class TrainingConfig:
    """Training runtime settings."""

    sample_index: Path
    output_root: Path
    run_id: str | None
    overwrite: bool
    batch_size: int
    num_workers: int
    epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    dtype: str
    amp: bool
    compile: bool
    seed: int
    log_every_steps: int
    save_every_steps: int
    preview_every_steps: int


@dataclass(frozen=True)
class InferenceConfig:
    """Inference runtime settings."""

    sample_index: Path
    checkpoint_path: Path | None
    output_root: Path
    batch_size: int
    num_workers: int
    compile: bool
    save_tensors: bool
    overwrite: bool


@dataclass(frozen=True)
class ScoringConfig:
    """Frame anomaly score composition settings."""

    topk_fraction: float
    topk_weight: float
    memory_distance_weight: float


@dataclass(frozen=True)
class RGBPatchPipelineConfig:
    """Resolved configuration for the single RGB-patch VAD pipeline."""

    dataset_name: str
    model: RGBPatchModelConfig
    flow_matching: FlowMatchingConfig
    training: TrainingConfig
    inference: InferenceConfig
    scoring: ScoringConfig
    raw: dict[str, Any]


@dataclass(frozen=True)
class RGBPatchShape:
    """Tensor shape metadata inferred from one RGB patch sample."""

    context_frames: int
    future_frames: int
    num_patches: int
    patch_dim: int
    image_size: int
    patch_size: int

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size


def load_pipeline_config(path: Path) -> RGBPatchPipelineConfig:
    """Load and validate a fixed-grid RGB patch pipeline configuration."""

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    configure_progress_from_config(raw)

    dataset = raw.get("dataset", {})
    model_raw = dict(raw.get("model", {}))
    memory_raw = dict(model_raw.get("memory", {}))
    training_raw = dict(raw.get("training", {}))
    inference_raw = dict(raw.get("inference", {}))
    flow_raw = dict(raw.get("flow_matching", {}))
    scoring_raw = dict(raw.get("scoring", {}))

    if str(model_raw.get("name", MODEL_NAME)) != MODEL_NAME:
        raise ValueError(f"model.name must be {MODEL_NAME}")
    model = RGBPatchModelConfig(
        image_size=int(model_raw.get("image_size", 256)),
        patch_size=int(model_raw.get("patch_size", 16)),
        hidden_size=int(model_raw.get("hidden_size", 512)),
        num_layers=int(model_raw.get("num_layers", 12)),
        num_heads=int(model_raw.get("num_heads", 8)),
        mlp_ratio=float(model_raw.get("mlp_ratio", 4.0)),
        dropout=float(model_raw.get("dropout", 0.0)),
        memory_size=int(memory_raw.get("size", 64)),
        memory_temperature=float(memory_raw.get("temperature", 0.25)),
        time_embedding_scale=float(model_raw.get("time_embedding_scale", 1000.0)),
    )
    flow_matching = FlowMatchingConfig(
        inference_steps=int(flow_raw.get("inference_steps", 8)),
        timestep_distribution=str(flow_raw.get("timestep_distribution", "uniform")),
    )
    training = TrainingConfig(
        sample_index=Path(
            training_raw.get(
                "sample_index", "data/02_processed/shanghai/metadata/samples_train.jsonl"
            )
        ),
        output_root=Path(training_raw.get("output_root", "checkpoints")),
        run_id=parse_optional_str(training_raw.get("run_id")),
        overwrite=bool(training_raw.get("overwrite", False)),
        batch_size=int(training_raw.get("batch_size", 4)),
        num_workers=int(training_raw.get("num_workers", 4)),
        epochs=int(training_raw.get("epochs", 1)),
        learning_rate=float(training_raw.get("learning_rate", 1.0e-4)),
        weight_decay=float(training_raw.get("weight_decay", 1.0e-2)),
        grad_clip_norm=float(training_raw.get("grad_clip_norm", 1.0)),
        dtype=str(training_raw.get("dtype", "float16")),
        amp=bool(training_raw.get("amp", True)),
        compile=bool(training_raw.get("compile", False)),
        seed=int(training_raw.get("seed", 42)),
        log_every_steps=int(training_raw.get("log_every_steps", 50)),
        save_every_steps=int(training_raw.get("save_every_steps", 1000)),
        preview_every_steps=int(training_raw.get("preview_every_steps", 5000)),
    )
    checkpoint_value = parse_optional_str(inference_raw.get("checkpoint_path"))
    inference = InferenceConfig(
        sample_index=Path(
            inference_raw.get(
                "sample_index", "data/02_processed/shanghai/metadata/samples_test.jsonl"
            )
        ),
        checkpoint_path=Path(checkpoint_value) if checkpoint_value else None,
        output_root=Path(inference_raw.get("output_root", "data/04_predictions/shanghai")),
        batch_size=int(inference_raw.get("batch_size", 4)),
        num_workers=int(inference_raw.get("num_workers", 4)),
        compile=bool(inference_raw.get("compile", False)),
        save_tensors=bool(inference_raw.get("save_tensors", False)),
        overwrite=bool(inference_raw.get("overwrite", False)),
    )
    scoring = ScoringConfig(
        topk_fraction=float(scoring_raw.get("topk_fraction", 0.10)),
        topk_weight=float(scoring_raw.get("topk_weight", 0.20)),
        memory_distance_weight=float(scoring_raw.get("memory_distance_weight", 0.0)),
    )
    validate_pipeline_config(model, flow_matching, training, inference, scoring)
    return RGBPatchPipelineConfig(
        dataset_name=str(dataset.get("name", "shanghai")),
        model=model,
        flow_matching=flow_matching,
        training=training,
        inference=inference,
        scoring=scoring,
        raw=raw,
    )


def parse_optional_str(value: Any) -> str | None:
    """Return a stripped optional string."""

    if value is None:
        return None
    parsed = str(value).strip()
    return parsed or None


def validate_pipeline_config(
    model: RGBPatchModelConfig,
    flow_matching: FlowMatchingConfig,
    training: TrainingConfig,
    inference: InferenceConfig,
    scoring: ScoringConfig,
) -> None:
    """Validate all configuration values before data or model construction."""

    if model.image_size <= 0:
        raise ValueError("model.image_size must be positive")
    if model.patch_size not in SUPPORTED_PATCH_SIZES:
        raise ValueError("model.patch_size must be one of: 16, 32")
    if model.image_size % model.patch_size:
        raise ValueError("model.image_size must be divisible by model.patch_size")
    if model.hidden_size <= 0 or model.num_layers <= 0 or model.num_heads <= 0:
        raise ValueError(
            "model.hidden_size, model.num_layers, and model.num_heads must be positive"
        )
    if model.hidden_size % model.num_heads:
        raise ValueError("model.hidden_size must be divisible by model.num_heads")
    if (
        model.mlp_ratio <= 0
        or model.memory_size <= 0
        or model.memory_temperature <= 0
        or model.time_embedding_scale <= 0
    ):
        raise ValueError(
            "model.mlp_ratio, model.memory settings, and time_embedding_scale must be positive"
        )
    if flow_matching.inference_steps <= 0:
        raise ValueError("flow_matching.inference_steps must be positive")
    if flow_matching.timestep_distribution != "uniform":
        raise ValueError("flow_matching.timestep_distribution must be uniform")
    if training.batch_size <= 0 or inference.batch_size <= 0:
        raise ValueError("training and inference batch sizes must be positive")
    if training.num_workers < 0 or inference.num_workers < 0:
        raise ValueError("training and inference num_workers must be non-negative")
    if training.epochs <= 0:
        raise ValueError("training.epochs must be positive")
    if training.learning_rate <= 0 or training.weight_decay < 0 or training.grad_clip_norm < 0:
        raise ValueError("training optimizer settings are invalid")
    if training.dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("training.dtype must be float16, bfloat16, or float32")
    if training.log_every_steps <= 0 or training.save_every_steps <= 0:
        raise ValueError("training log/save intervals must be positive")
    if training.preview_every_steps < 0:
        raise ValueError("training.preview_every_steps must be non-negative")
    if not 0 < scoring.topk_fraction <= 1:
        raise ValueError("scoring.topk_fraction must be in (0, 1]")
    if scoring.topk_weight < 0 or scoring.memory_distance_weight < 0:
        raise ValueError("scoring weights must be non-negative")


def load_rgb_frame(path: str | Path, image_size: int) -> torch.Tensor:
    """Load one frame as RGB tensor in ``[-1, 1]`` with square resize."""

    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
        values = np.asarray(rgb, dtype=np.float32).copy()
    return torch.from_numpy(values).permute(2, 0, 1).div(127.5).sub(1.0)


def patchify_rgb(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Patchify RGB frames in deterministic row-major order.

    ``images`` is shaped ``[..., 3, H, W]`` and the output is
    ``[..., P, 3 * patch_size * patch_size]``.  The function is exactly
    inverted by :func:`unpatchify_rgb`.
    """

    if images.ndim < 4 or images.shape[-3] != RGB_CHANNELS:
        raise ValueError("images must be shaped [..., 3, H, W]")
    if patch_size not in SUPPORTED_PATCH_SIZES:
        raise ValueError("patch_size must be one of: 16, 32")
    height, width = (int(images.shape[-2]), int(images.shape[-1]))
    if height != width or height % patch_size or width % patch_size:
        raise ValueError("RGB image height and width must be square and divisible by patch_size")

    leading_shape = images.shape[:-3]
    grid_size = height // patch_size
    flat = images.reshape(-1, RGB_CHANNELS, height, width)
    patches = flat.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.reshape(-1, grid_size * grid_size, RGB_CHANNELS * patch_size * patch_size)
    return patches.reshape(*leading_shape, grid_size * grid_size, -1)


def unpatchify_rgb(
    patches: torch.Tensor,
    *,
    image_size: int,
    patch_size: int,
) -> torch.Tensor:
    """Invert row-major RGB patchification exactly."""

    if patches.ndim < 3:
        raise ValueError("patches must be shaped [..., P, patch_dim]")
    if patch_size not in SUPPORTED_PATCH_SIZES:
        raise ValueError("patch_size must be one of: 16, 32")
    if image_size <= 0 or image_size % patch_size:
        raise ValueError("image_size must be divisible by patch_size")

    grid_size = image_size // patch_size
    expected_patches = grid_size * grid_size
    expected_dim = RGB_CHANNELS * patch_size * patch_size
    if patches.shape[-2:] != (expected_patches, expected_dim):
        raise ValueError(
            "patch shape does not match image_size and patch_size: "
            f"got {tuple(patches.shape[-2:])}, expected {(expected_patches, expected_dim)}"
        )

    leading_shape = patches.shape[:-2]
    flat = patches.reshape(-1, grid_size, grid_size, RGB_CHANNELS, patch_size, patch_size)
    frames = flat.permute(0, 3, 1, 4, 2, 5).contiguous()
    frames = frames.reshape(-1, RGB_CHANNELS, image_size, image_size)
    return frames.reshape(*leading_shape, RGB_CHANNELS, image_size, image_size)


class RGBPatchDataset(Dataset):
    """Load past/future RGB frame windows and fixed-grid patchify both."""

    def __init__(self, records: list[dict[str, Any]], model: RGBPatchModelConfig) -> None:
        if not records:
            raise ValueError("RGBPatchDataset requires at least one record")
        self.records = records
        self.model = model

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        context_paths = required_frame_paths(record, "context_frame_paths")
        future_paths = required_frame_paths(record, "future_frame_paths")
        context = torch.stack(
            [load_rgb_frame(path, self.model.image_size) for path in context_paths]
        )
        target = torch.stack([load_rgb_frame(path, self.model.image_size) for path in future_paths])
        return {
            "context": patchify_rgb(context, self.model.patch_size),
            "target": patchify_rgb(target, self.model.patch_size),
            "metadata": record,
        }


def required_frame_paths(record: dict[str, Any], key: str) -> list[str]:
    """Return a non-empty string frame-path list from a sample record."""

    paths = record.get(key)
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"sample record must contain a non-empty {key}")
    return [str(path) for path in paths]


def patch_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack fixed-shape patch tensors and preserve sample metadata."""

    context_shapes = {tuple(sample["context"].shape) for sample in batch}
    target_shapes = {tuple(sample["target"].shape) for sample in batch}
    if len(context_shapes) != 1 or len(target_shapes) != 1:
        raise ValueError("all samples in a batch must have equal context and target patch shapes")
    return {
        "context": torch.stack([sample["context"] for sample in batch]),
        "target": torch.stack([sample["target"] for sample in batch]),
        "metadata": [sample["metadata"] for sample in batch],
    }


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    """Create the shared RGB-patch DataLoader."""

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": patch_collate,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def load_sample_records(
    path: Path,
    *,
    normal_only: bool,
    limit_samples: int | None,
) -> list[dict[str, Any]]:
    """Load processed-window records used directly for training or inference."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing processed sample index: {path}")
    records = read_jsonl(path)
    if normal_only:
        records = [record for record in records if int(record.get("future_label", 0)) == 0]
    if limit_samples is not None:
        if limit_samples <= 0:
            raise ValueError("limit_samples must be positive")
        records = records[:limit_samples]
    if not records:
        raise ValueError(f"No sample records available from {path}")
    for record in records:
        required_frame_paths(record, "context_frame_paths")
        required_frame_paths(record, "future_frame_paths")
    return records


def infer_patch_shape(sample: dict[str, Any], model: RGBPatchModelConfig) -> RGBPatchShape:
    """Infer and validate DiT shape metadata from one dataset sample."""

    context = sample["context"]
    target = sample["target"]
    if context.ndim != 3 or target.ndim != 3:
        raise ValueError("context and target must be shaped [T, P, patch_dim]")
    expected_patches = (model.image_size // model.patch_size) ** 2
    expected_dim = RGB_CHANNELS * model.patch_size * model.patch_size
    if context.shape[1:] != (expected_patches, expected_dim):
        raise ValueError("context patch shape does not match configured fixed grid")
    if target.shape[1:] != (expected_patches, expected_dim):
        raise ValueError("future target patch shape does not match configured fixed grid")
    return RGBPatchShape(
        context_frames=int(context.shape[0]),
        future_frames=int(target.shape[0]),
        num_patches=expected_patches,
        patch_dim=expected_dim,
        image_size=model.image_size,
        patch_size=model.patch_size,
    )


def score_patch_prediction(
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    scoring: ScoringConfig,
    memory_distance: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return full-frame, top-k, and optional-memory anomaly score components."""

    if target.shape != prediction.shape or target.ndim != 4:
        raise ValueError("target and prediction must both be shaped [B, T, P, patch_dim]")
    patch_error = (target.float() - prediction.float()).square().mean(dim=-1)
    full_frame = patch_error.mean(dim=-1)
    topk_count = max(1, math.ceil(patch_error.shape[-1] * scoring.topk_fraction))
    topk = patch_error.topk(topk_count, dim=-1).values.mean(dim=-1)
    score = full_frame + scoring.topk_weight * topk
    memory = None
    if memory_distance is not None:
        if memory_distance.shape != (target.shape[0],):
            raise ValueError("memory_distance must be shaped [B]")
        memory = memory_distance[:, None].expand_as(full_frame)
        score = score + scoring.memory_distance_weight * memory
    return {
        "patch_error": patch_error,
        "full_frame": full_frame,
        "topk": topk,
        "memory_distance": memory if memory is not None else torch.zeros_like(full_frame),
        "score": score,
    }


def aggregate_frame_scores(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average overlapping sample predictions into unique video-frame scores."""

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        for index, frame_idx in enumerate(record["future_frames"]):
            key = (str(record["video_id"]), int(frame_idx))
            grouped.setdefault(key, []).append(
                {
                    "label": int(record["future_frame_labels"][index]),
                    "score": float(record["scores"][index]),
                    "full_frame": float(record["full_frame_scores"][index]),
                    "topk": float(record["topk_scores"][index]),
                    "memory_distance": float(record["memory_distances"][index]),
                }
            )

    output = []
    for (video_id, frame_idx), values in sorted(grouped.items()):
        labels = {value["label"] for value in values}
        if len(labels) != 1:
            raise ValueError(f"Conflicting labels for {video_id} frame {frame_idx}")
        output.append(
            {
                "video_id": video_id,
                "frame_idx": frame_idx,
                "label": labels.pop(),
                "num_votes": len(values),
                "score": float(np.mean([value["score"] for value in values])),
                "full_frame_score": float(np.mean([value["full_frame"] for value in values])),
                "topk_score": float(np.mean([value["topk"] for value in values])),
                "memory_distance": float(np.mean([value["memory_distance"] for value in values])),
            }
        )
    return output


def compute_binary_metrics(frame_scores: list[dict[str, Any]]) -> dict[str, float | None]:
    """Compute frame-level ROC-AUC/AP when labels contain both classes."""

    labels = [int(record["label"]) for record in frame_scores]
    scores = [float(record["score"]) for record in frame_scores]
    if len(set(labels)) < 2:
        return {"roc_auc": None, "average_precision": None}
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file."""

    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write UTF-8 JSONL records, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def config_to_dict(config: RGBPatchPipelineConfig) -> dict[str, Any]:
    """Serialize resolved runtime configuration for artifacts and checkpoints."""

    value = asdict(config)
    return convert_paths(value)


def convert_paths(value: Any) -> Any:
    """Recursively convert paths to strings for YAML/JSON serialization."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: convert_paths(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [convert_paths(item) for item in value]
    if isinstance(value, list):
        return [convert_paths(item) for item in value]
    return value
