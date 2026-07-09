"""Shared config, dataset, and scoring utilities for Z DiT pipelines."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.models import (
    CONDITION_MODE_APPEARANCE_ONLY,
    CONDITION_MODE_APPEARANCE_TRACK,
    CONDITION_MODE_BASELINE,
    CONDITION_MODE_TRACK_ONLY,
    ConditionedZDiT,
    FlowMatchingSampler,
    ZDiTShape,
    condition_mode_uses_appearance,
    condition_mode_uses_track,
    normalize_condition_mode,
)
from src.pipelines.feature_eng_pipeline import read_jsonl, write_jsonl
from src.pipelines.object_track_pipeline import TRACK_CHANNELS, load_track_feature_payload
from src.utils import configure_progress_from_config, progress_bar

PREDICTION_VELOCITY = "velocity"
SCORE_NORMALIZED_Z_MSE = "normalized_z_mse"
SCORE_Z_MSE = "z_mse"
SCORE_VARIANT_GLOBAL = "global"
SCORE_VARIANT_PATCH = "patch"
SCORE_VARIANT_LOW_WEIGHTED = "low_weighted"
SCORE_VARIANT_GLOBAL_PATCH = "global_patch"
SCORE_VARIANT_MOTION_PATCH = "motion_patch"
SCORE_VARIANT_MOTION_GLOBAL_PATCH = "motion_global_patch"
SCORE_VARIANT_TRACK_WEIGHTED = "track_weighted"
SCORE_VARIANT_TRACK_REGION_TOPK = "track_region_topk"
SCORE_VARIANT_MEAN_TOPK_PLUS_TRACK_REGION = "mean_topk_plus_track_region"
SCORE_PRIMARY_RAW = "raw"
SCORE_PRIMARY_VIDEO_CENTERED = "video_centered"
SCORE_PRIMARY_ROLLING_CENTERED = "rolling_centered"
SCORE_PRIMARY_SCENE_CALIBRATED = "scene_calibrated"
HIGH_TOKEN_REDUCTIONS = {"uniform", "foreground_background", "aligned_grid"}
MODEL_NAME = "high_low_conditioned_z_dit"
CONDITION_MODES = {
    CONDITION_MODE_BASELINE,
    CONDITION_MODE_APPEARANCE_ONLY,
    CONDITION_MODE_TRACK_ONLY,
    CONDITION_MODE_APPEARANCE_TRACK,
}
TRACK_ABLATION_REAL = "real"
TRACK_ABLATION_ZERO = "zero"
TRACK_ABLATION_SHUFFLED = "shuffled"
TRACK_ABLATION_ALIASES = {
    "real": TRACK_ABLATION_REAL,
    "real_track": TRACK_ABLATION_REAL,
    "zero": TRACK_ABLATION_ZERO,
    "zero_track": TRACK_ABLATION_ZERO,
    "shuffled": TRACK_ABLATION_SHUFFLED,
    "shuffled_track": TRACK_ABLATION_SHUFFLED,
}
TRACK_CHANNEL_MASKS = {
    "all_channels": TRACK_CHANNELS,
    "objectness_only": ("objectness",),
    "speed_only": ("speed",),
    "trajectory_only": ("trajectory",),
    "objectness_speed": ("objectness", "speed"),
    "objectness_trajectory": ("objectness", "trajectory"),
    "speed_trajectory": ("speed", "trajectory"),
}
SCORE_VARIANTS = {
    SCORE_VARIANT_GLOBAL,
    SCORE_VARIANT_PATCH,
    SCORE_VARIANT_LOW_WEIGHTED,
    SCORE_VARIANT_GLOBAL_PATCH,
    SCORE_VARIANT_MOTION_PATCH,
    SCORE_VARIANT_MOTION_GLOBAL_PATCH,
    SCORE_VARIANT_TRACK_WEIGHTED,
    SCORE_VARIANT_TRACK_REGION_TOPK,
    SCORE_VARIANT_MEAN_TOPK_PLUS_TRACK_REGION,
}


@dataclass(frozen=True)
class DiTConfig:
    """Flow transformer architecture settings."""

    hidden_size: int = 512
    num_layers: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.05
    gradient_checkpointing: bool = False


@dataclass(frozen=True)
class HighAdapterConfig:
    """C_high token adapter settings."""

    input_dim: str = "auto"
    max_tokens: int | None = 512
    token_reduction: str = "uniform"
    foreground_ratio: float = 0.75
    use_pos_embedding: bool = True


@dataclass(frozen=True)
class ZAdapterConfig:
    """Z token adapter settings."""

    patch_size: int = 2
    use_temporal_pos: bool = True
    use_spatial_pos: bool = True


@dataclass(frozen=True)
class LowAdapterConfig:
    """Simple C_low token adapter settings."""

    patch_size: int = 16
    use_temporal_pos: bool = True
    use_spatial_pos: bool = True


@dataclass(frozen=True)
class TrackGridConditionConfig:
    """Inference-time C_track ablation controls."""

    ablation_mode: str = TRACK_ABLATION_REAL
    shuffle_seed: int = 0
    channel_mask: str = "all_channels"
    gate_override: float | None = None


@dataclass(frozen=True)
class ConditionConfig:
    """Optional model condition switches."""

    mode: str = CONDITION_MODE_BASELINE
    use_track_grid: bool = False
    track_grid_gate: float = 0.1
    track_grid: TrackGridConditionConfig = field(default_factory=TrackGridConditionConfig)


@dataclass(frozen=True)
class ModelConfig:
    """Z DiT model config."""

    name: str
    prediction_type: str
    dit: DiTConfig
    high_adapter: HighAdapterConfig
    z_adapter: ZAdapterConfig
    low_adapter: LowAdapterConfig
    condition: ConditionConfig


def condition_uses_patch_condition(condition: ConditionConfig) -> bool:
    """Return whether config selects the simplified patch-condition backbone."""

    return normalize_condition_mode(condition.mode) != CONDITION_MODE_BASELINE


def condition_uses_appearance_features(condition: ConditionConfig) -> bool:
    """Return whether C_low/appearance grids are required by the model."""

    return condition_mode_uses_appearance(condition.mode)


def condition_uses_track_features(condition: ConditionConfig) -> bool:
    """Return whether dense track grids are required by the model."""

    return condition_mode_uses_track(condition.mode) or condition_uses_legacy_track_gate(condition)


def condition_uses_legacy_track_gate(condition: ConditionConfig) -> bool:
    """Return whether the old separate C_track token gate is active."""

    return normalize_condition_mode(condition.mode) == CONDITION_MODE_BASELINE and bool(
        condition.use_track_grid
    )


@dataclass(frozen=True)
class FlowMatchingConfig:
    """Conditional flow matching config."""

    inference_steps: int
    timestep_distribution: str
    time_embedding_scale: float
    normalize_z: bool
    beta_alpha: float
    beta_beta: float
    beta_s: float


@dataclass(frozen=True)
class LossConfig:
    """Training loss config."""

    alpha: float = 0.2
    topk_fraction: float = 0.1


@dataclass(frozen=True)
class OptimizationConfig:
    """Backend optimization settings with hardware/runtime tradeoffs."""

    matmul_precision: str = "high"
    enable_tf32: bool = True
    cudnn_benchmark: bool = True
    enable_flash_sdp: bool = True
    fused_adamw: str = "auto"
    prefetch_factor: int = 4


@dataclass(frozen=True)
class TrainingConfig:
    """Training runtime config."""

    feature_index: Path
    init_checkpoint_path: Path | None
    output_root: Path
    run_id: str | None
    overwrite: bool
    batch_size: int
    num_workers: int
    max_steps: int
    val_fraction: float
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    dtype: str
    amp: bool
    compile: bool
    seed: int
    log_every_steps: int
    save_every_steps: int


@dataclass(frozen=True)
class InferenceConfig:
    """Inference runtime config."""

    feature_index: Path
    checkpoint_path: Path | None
    output_root: Path
    batch_size: int
    num_workers: int
    mode: str
    compile: bool
    save_tensors: bool
    overwrite: bool


@dataclass(frozen=True)
class NormalizedZMSEConfig:
    """Normalized Z MSE settings."""

    eps: float = 1.0e-6


@dataclass(frozen=True)
class DecodedMSEConfig:
    """Decoded MSE settings reserved for a later heavier scoring mode."""

    enabled: bool = False
    alpha: float = 0.1


@dataclass(frozen=True)
class ScoreNormalizationConfig:
    """Frame-score centering settings."""

    enabled: bool = True
    primary: str = SCORE_PRIMARY_VIDEO_CENTERED
    rolling_window: int = 128
    rolling_min_history: int = 16
    rolling_windows: tuple[int, ...] = ()


@dataclass(frozen=True)
class ScoreSweepConfig:
    """Additional score variants evaluated without retraining."""

    betas: tuple[float, ...] = ()
    topk_fractions: tuple[float, ...] = ()


@dataclass(frozen=True)
class ScoreCalibrationConfig:
    """Scene/video normal score calibration settings."""

    enabled: bool = False
    stats_path: Path | None = None
    fit_output_path: Path | None = None
    group_key: str = "scene_id"
    eps: float = 1.0e-6


@dataclass(frozen=True)
class ScoringConfig:
    """Anomaly scoring config."""

    method: str
    frame_aggregation: str
    variant: str
    variants: tuple[str, ...]
    topk_fraction: float
    low_weight_alpha: float
    low_weight_eps: float
    beta: float
    motion_topk_fraction: float
    track_weight_alpha: float
    track_speed_alpha: float
    track_trajectory_alpha: float
    track_weight_max: float
    track_region_beta: float
    normalized_z_mse: NormalizedZMSEConfig
    decoded_mse: DecodedMSEConfig
    score_normalization: ScoreNormalizationConfig
    sweep: ScoreSweepConfig
    calibration: ScoreCalibrationConfig


@dataclass(frozen=True)
class ZDiTPipelineConfig:
    """Resolved training/inference config for Z DiT pipelines."""

    dataset_name: str
    model: ModelConfig
    flow_matching: FlowMatchingConfig
    loss: LossConfig
    optimization: OptimizationConfig
    training: TrainingConfig
    inference: InferenceConfig
    scoring: ScoringConfig
    raw: dict[str, Any]


class FeatureZDataset(Dataset):
    """Dataset loading cached C_high, optional C_low, and Z tensors."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        require_track_features: bool = False,
        require_low_features: bool = False,
    ) -> None:
        if not records:
            raise ValueError("FeatureZDataset requires at least one record")
        self.records = records
        self.require_track_features = require_track_features
        self.require_low_features = require_low_features

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        high_payload = load_torch_payload(Path(record["high_feature_path"]))
        z_payload = load_torch_payload(Path(record["z_path"]))
        low = None
        low_path = record.get("low_feature_path")
        if low_path is not None:
            low_payload = load_torch_payload(Path(low_path))
            low = low_payload["low"].float()
        elif self.require_low_features:
            raise ValueError(
                "condition.mode requires feature records to contain low_feature_path: "
                f"sample_id={record.get('sample_id')}"
            )
        track = None
        track_path = record.get("track_feature_path")
        if track_path is not None:
            track_payload = load_track_feature_payload(Path(track_path))
            track = {
                "context_grid": track_payload["context_grid"].float(),
                "future_grid": track_payload["future_grid"].float(),
                "metadata": track_payload["metadata"],
            }
        elif self.require_track_features:
            raise ValueError(
                "condition.use_track_grid=true requires feature records to contain "
                f"track_feature_path: sample_id={record.get('sample_id')}"
            )
        return {
            "high": high_payload["high"].float(),
            "z": z_payload["z"].float(),
            "low": low,
            "track": track,
            "metadata": record,
        }


def load_torch_payload(path: Path) -> dict[str, Any]:
    """Load one local torch payload."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing feature tensor: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def feature_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate feature tensors while preserving metadata."""

    low_tensors = [sample["low"] for sample in batch]
    has_low = [low is not None for low in low_tensors]
    if any(has_low) and not all(has_low):
        raise ValueError("Feature batch mixes records with and without low features")
    track_payloads = [sample["track"] for sample in batch]
    has_track = [track is not None for track in track_payloads]
    if any(has_track) and not all(has_track):
        raise ValueError("Feature batch mixes records with and without track features")
    track = None
    if all(has_track):
        context_shapes = {tuple(sample["track"]["context_grid"].shape) for sample in batch}
        future_shapes = {tuple(sample["track"]["future_grid"].shape) for sample in batch}
        if len(context_shapes) != 1 or len(future_shapes) != 1:
            raise ValueError("Track context_grid and future_grid shapes must match within a batch")
        track = {
            "context_grid": torch.stack([sample["track"]["context_grid"] for sample in batch]),
            "future_grid": torch.stack([sample["track"]["future_grid"] for sample in batch]),
            "metadata": [sample["track"]["metadata"] for sample in batch],
        }

    return {
        "high": torch.stack([sample["high"] for sample in batch]),
        "z": torch.stack([sample["z"] for sample in batch]),
        "low": torch.stack(low_tensors) if all(has_low) else None,
        "track": track,
        "metadata": [sample["metadata"] for sample in batch],
    }


def load_pipeline_config(path: Path) -> ZDiTPipelineConfig:
    """Load and validate Z DiT train/inference config."""

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    configure_progress_from_config(raw)
    dataset = raw.get("dataset", {})
    model = load_model_config(raw.get("model", {}))
    flow_matching = load_flow_matching_config(raw.get("flow_matching", {}))
    loss = load_loss_config(raw.get("loss", {}))
    optimization = load_optimization_config(raw.get("optimization", {}))
    training = load_training_config(raw.get("training", {}))
    inference = load_inference_config(raw.get("inference", {}))
    scoring = load_scoring_config(raw.get("scoring", {}))
    validate_pipeline_config(model, flow_matching, loss, optimization, training, inference, scoring)

    return ZDiTPipelineConfig(
        dataset_name=str(dataset.get("name", "shanghaitech")),
        model=model,
        flow_matching=flow_matching,
        loss=loss,
        optimization=optimization,
        training=training,
        inference=inference,
        scoring=scoring,
        raw=raw,
    )


def load_model_config(config: dict[str, Any]) -> ModelConfig:
    """Resolve model config."""

    dit = dict(config.get("dit", {}))
    high = dict(config.get("high_adapter", {}))
    z = dict(config.get("z_adapter", {}))
    low = dict(config.get("low_adapter", {}))
    condition = dict(config.get("condition", {}))
    track_grid = dict(condition.get("track_grid", {}))
    name = str(config.get("name", MODEL_NAME))
    return ModelConfig(
        name=name,
        prediction_type=str(config.get("prediction_type", PREDICTION_VELOCITY)),
        dit=DiTConfig(
            hidden_size=int(dit.get("hidden_size", 512)),
            num_layers=int(dit.get("num_layers", 12)),
            num_heads=int(dit.get("num_heads", 8)),
            mlp_ratio=float(dit.get("mlp_ratio", 4.0)),
            dropout=float(dit.get("dropout", 0.05)),
            gradient_checkpointing=bool(dit.get("gradient_checkpointing", False)),
        ),
        high_adapter=HighAdapterConfig(
            input_dim=str(high.get("input_dim", "auto")),
            max_tokens=parse_optional_int(high.get("max_tokens", 512)),
            token_reduction=str(high.get("token_reduction", "uniform")),
            foreground_ratio=float(high.get("foreground_ratio", 0.75)),
            use_pos_embedding=bool(high.get("use_pos_embedding", True)),
        ),
        z_adapter=ZAdapterConfig(
            patch_size=int(z.get("patch_size", 2)),
            use_temporal_pos=bool(z.get("use_temporal_pos", True)),
            use_spatial_pos=bool(z.get("use_spatial_pos", True)),
        ),
        low_adapter=LowAdapterConfig(
            patch_size=int(low.get("patch_size", 16)),
            use_temporal_pos=bool(low.get("use_temporal_pos", True)),
            use_spatial_pos=bool(low.get("use_spatial_pos", True)),
        ),
        condition=ConditionConfig(
            mode=normalize_condition_mode(condition.get("mode", CONDITION_MODE_BASELINE)),
            use_track_grid=bool(condition.get("use_track_grid", False)),
            track_grid_gate=float(condition.get("track_grid_gate", 0.1)),
            track_grid=TrackGridConditionConfig(
                ablation_mode=normalize_track_ablation_mode(
                    str(track_grid.get("ablation_mode", TRACK_ABLATION_REAL))
                ),
                shuffle_seed=int(track_grid.get("shuffle_seed", 0)),
                channel_mask=str(track_grid.get("channel_mask", "all_channels")),
                gate_override=parse_optional_float(track_grid.get("gate_override")),
            ),
        ),
    )


def load_flow_matching_config(config: dict[str, Any]) -> FlowMatchingConfig:
    """Resolve flow matching config."""

    return FlowMatchingConfig(
        inference_steps=int(config.get("inference_steps", 4)),
        timestep_distribution=str(config.get("timestep_distribution", "gr00t_beta")),
        time_embedding_scale=float(config.get("time_embedding_scale", 1000.0)),
        normalize_z=bool(config.get("normalize_z", True)),
        beta_alpha=float(config.get("beta_alpha", 1.5)),
        beta_beta=float(config.get("beta_beta", 1.0)),
        beta_s=float(config.get("beta_s", 0.999)),
    )


def load_loss_config(config: dict[str, Any]) -> LossConfig:
    """Resolve DiT training loss config."""

    return LossConfig(
        alpha=float(config.get("alpha", 0.2)),
        topk_fraction=float(config.get("topk_fraction", 0.1)),
    )


def load_optimization_config(config: dict[str, Any]) -> OptimizationConfig:
    """Resolve backend optimization config."""

    return OptimizationConfig(
        matmul_precision=str(config.get("matmul_precision", "high")),
        enable_tf32=bool(config.get("enable_tf32", True)),
        cudnn_benchmark=bool(config.get("cudnn_benchmark", True)),
        enable_flash_sdp=bool(config.get("enable_flash_sdp", True)),
        fused_adamw=str(config.get("fused_adamw", "auto")).lower(),
        prefetch_factor=int(config.get("prefetch_factor", 4)),
    )


def load_training_config(config: dict[str, Any]) -> TrainingConfig:
    """Resolve training config."""

    return TrainingConfig(
        feature_index=Path(
            config.get("feature_index", "data/03_features/shanghaitech/train_feature_index.jsonl")
        ),
        init_checkpoint_path=parse_optional_path(config.get("init_checkpoint_path")),
        output_root=Path(config.get("output_root", "checkpoints")),
        run_id=parse_optional_str(config.get("run_id")),
        overwrite=bool(config.get("overwrite", False)),
        batch_size=int(config.get("batch_size", 4)),
        num_workers=int(config.get("num_workers", 4)),
        max_steps=int(config.get("max_steps", 20000)),
        val_fraction=float(config.get("val_fraction", 0.05)),
        learning_rate=float(config.get("learning_rate", 1.0e-4)),
        weight_decay=float(config.get("weight_decay", 1.0e-2)),
        grad_clip_norm=float(config.get("grad_clip_norm", 1.0)),
        dtype=str(config.get("dtype", "float16")),
        amp=bool(config.get("amp", True)),
        compile=bool(config.get("compile", False)),
        seed=int(config.get("seed", 42)),
        log_every_steps=int(config.get("log_every_steps", 50)),
        save_every_steps=int(config.get("save_every_steps", 1000)),
    )


def load_inference_config(config: dict[str, Any]) -> InferenceConfig:
    """Resolve inference config."""

    checkpoint = parse_optional_str(config.get("checkpoint_path"))
    return InferenceConfig(
        feature_index=Path(
            config.get("feature_index", "data/03_features/shanghaitech/test_feature_index.jsonl")
        ),
        checkpoint_path=Path(checkpoint) if checkpoint else None,
        output_root=Path(config.get("output_root", "data/04_predictions/shanghaitech")),
        batch_size=int(config.get("batch_size", 4)),
        num_workers=int(config.get("num_workers", 4)),
        mode=str(config.get("mode", "offline_cached")),
        compile=bool(config.get("compile", False)),
        save_tensors=bool(config.get("save_tensors", False)),
        overwrite=bool(config.get("overwrite", False)),
    )


def load_scoring_config(config: dict[str, Any]) -> ScoringConfig:
    """Resolve scoring config."""

    normalized = dict(config.get("normalized_z_mse", {}))
    decoded = dict(config.get("decoded_mse", {}))
    score_normalization = dict(config.get("score_normalization", {}))
    sweep = dict(config.get("sweep", {}))
    calibration = dict(config.get("calibration", {}))
    variants_value = config.get(
        "variants",
        [
            SCORE_VARIANT_GLOBAL,
            SCORE_VARIANT_PATCH,
            SCORE_VARIANT_LOW_WEIGHTED,
            SCORE_VARIANT_GLOBAL_PATCH,
            SCORE_VARIANT_MOTION_PATCH,
            SCORE_VARIANT_MOTION_GLOBAL_PATCH,
        ],
    )
    if isinstance(variants_value, str):
        variants = (str(variants_value),)
    else:
        variants = tuple(str(value) for value in variants_value)
    return ScoringConfig(
        method=str(config.get("method", SCORE_NORMALIZED_Z_MSE)),
        frame_aggregation=str(config.get("frame_aggregation", "mean")),
        variant=str(config.get("variant", SCORE_VARIANT_GLOBAL_PATCH)),
        variants=variants,
        topk_fraction=float(config.get("topk_fraction", 0.10)),
        low_weight_alpha=float(config.get("low_weight_alpha", 1.0)),
        low_weight_eps=float(config.get("low_weight_eps", 1.0e-6)),
        beta=float(config.get("beta", 0.2)),
        motion_topk_fraction=float(config.get("motion_topk_fraction", 0.25)),
        track_weight_alpha=float(config.get("track_weight_alpha", 1.0)),
        track_speed_alpha=float(config.get("track_speed_alpha", 1.0)),
        track_trajectory_alpha=float(config.get("track_trajectory_alpha", 0.5)),
        track_weight_max=float(config.get("track_weight_max", 5.0)),
        track_region_beta=float(config.get("track_region_beta", config.get("beta", 0.2))),
        normalized_z_mse=NormalizedZMSEConfig(
            eps=float(normalized.get("eps", 1.0e-6)),
        ),
        decoded_mse=DecodedMSEConfig(
            enabled=bool(decoded.get("enabled", False)),
            alpha=float(decoded.get("alpha", 0.1)),
        ),
        score_normalization=ScoreNormalizationConfig(
            enabled=bool(score_normalization.get("enabled", True)),
            primary=str(score_normalization.get("primary", SCORE_PRIMARY_VIDEO_CENTERED)),
            rolling_window=int(score_normalization.get("rolling_window", 128)),
            rolling_min_history=int(score_normalization.get("rolling_min_history", 16)),
            rolling_windows=parse_int_tuple(score_normalization.get("rolling_windows", ())),
        ),
        sweep=ScoreSweepConfig(
            betas=parse_float_tuple(sweep.get("betas", ())),
            topk_fractions=parse_float_tuple(sweep.get("topk_fractions", ())),
        ),
        calibration=ScoreCalibrationConfig(
            enabled=bool(calibration.get("enabled", False)),
            stats_path=parse_optional_path(calibration.get("stats_path")),
            fit_output_path=parse_optional_path(calibration.get("fit_output_path")),
            group_key=str(calibration.get("group_key", "scene_id")),
            eps=float(calibration.get("eps", 1.0e-6)),
        ),
    )


def validate_pipeline_config(
    model: ModelConfig,
    flow_matching: FlowMatchingConfig,
    loss: LossConfig,
    optimization: OptimizationConfig,
    training: TrainingConfig,
    inference: InferenceConfig,
    scoring: ScoringConfig,
) -> None:
    """Validate resolved config."""

    if model.name != MODEL_NAME:
        raise ValueError(f"model.name must be {MODEL_NAME}")
    if model.prediction_type != PREDICTION_VELOCITY:
        raise ValueError("Only velocity prediction is supported")
    if model.dit.hidden_size <= 0 or model.dit.num_layers <= 0 or model.dit.num_heads <= 0:
        raise ValueError("model.dit hidden_size, num_layers, and num_heads must be positive")
    if model.dit.hidden_size % model.dit.num_heads:
        raise ValueError("model.dit.hidden_size must be divisible by model.dit.num_heads")
    if model.high_adapter.token_reduction not in HIGH_TOKEN_REDUCTIONS:
        raise ValueError(
            "high_adapter.token_reduction must be uniform, foreground_background, or aligned_grid"
        )
    if model.high_adapter.max_tokens is not None and model.high_adapter.max_tokens <= 0:
        raise ValueError("model.high_adapter.max_tokens must be positive or null")
    if not 0 < model.high_adapter.foreground_ratio <= 1:
        raise ValueError("model.high_adapter.foreground_ratio must be in (0, 1]")
    if model.z_adapter.patch_size <= 0:
        raise ValueError("model.z_adapter.patch_size must be positive")
    if model.low_adapter.patch_size <= 0:
        raise ValueError("model.low_adapter.patch_size must be positive")
    normalize_condition_mode(model.condition.mode)
    if model.condition.track_grid_gate < 0:
        raise ValueError("model.condition.track_grid_gate must be non-negative")
    normalize_track_ablation_mode(model.condition.track_grid.ablation_mode)
    normalize_track_channel_names(model.condition.track_grid.channel_mask)
    if model.condition.track_grid.gate_override is not None:
        if model.condition.track_grid.gate_override < 0:
            raise ValueError("model.condition.track_grid.gate_override must be non-negative")
        if not condition_uses_legacy_track_gate(model.condition):
            raise ValueError(
                "model.condition.track_grid.gate_override is only supported for "
                "condition.mode=baseline with condition.use_track_grid=true"
            )
    if flow_matching.inference_steps <= 0:
        raise ValueError("flow_matching.inference_steps must be positive")
    if flow_matching.timestep_distribution not in {"uniform", "gr00t_beta"}:
        raise ValueError("flow_matching.timestep_distribution must be uniform or gr00t_beta")
    if flow_matching.time_embedding_scale <= 0:
        raise ValueError("flow_matching.time_embedding_scale must be positive")
    if (
        flow_matching.beta_alpha <= 0
        or flow_matching.beta_beta <= 0
        or not 0 < flow_matching.beta_s <= 1
    ):
        raise ValueError("flow_matching beta parameters must be positive with beta_s in (0, 1]")
    if loss.alpha < 0:
        raise ValueError("loss.alpha must be non-negative")
    if not 0 < loss.topk_fraction <= 1:
        raise ValueError("loss.topk_fraction must be in (0, 1]")
    if optimization.matmul_precision not in {"highest", "high", "medium"}:
        raise ValueError("optimization.matmul_precision must be highest, high, or medium")
    if optimization.fused_adamw not in {"auto", "true", "false"}:
        raise ValueError("optimization.fused_adamw must be auto, true, or false")
    if optimization.prefetch_factor <= 0:
        raise ValueError("optimization.prefetch_factor must be positive")
    if training.batch_size <= 0 or inference.batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if training.num_workers < 0 or inference.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if training.max_steps <= 0:
        raise ValueError("training.max_steps must be positive")
    if not 0 <= training.val_fraction < 1:
        raise ValueError("training.val_fraction must be in [0, 1)")
    if training.dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("training.dtype must be float16, bfloat16, or float32")
    if inference.mode != "offline_cached":
        raise ValueError("Only inference.mode='offline_cached' is supported")
    if scoring.method not in {SCORE_NORMALIZED_Z_MSE, SCORE_Z_MSE}:
        raise ValueError("scoring.method must be normalized_z_mse or z_mse")
    if scoring.frame_aggregation != "mean":
        raise ValueError("Only scoring.frame_aggregation='mean' is supported")
    if not scoring.variants:
        raise ValueError("scoring.variants must contain at least one score variant")
    unknown_variants = [variant for variant in scoring.variants if variant not in SCORE_VARIANTS]
    if unknown_variants:
        raise ValueError(
            "scoring.variants must use global, patch, low_weighted, global_patch, "
            "motion_patch, motion_global_patch, track_weighted, track_region_topk, "
            "or mean_topk_plus_track_region: "
            f"{unknown_variants}"
        )
    if scoring.variant not in SCORE_VARIANTS:
        raise ValueError(
            "scoring.variant must be global, patch, low_weighted, global_patch, "
            "motion_patch, motion_global_patch, track_weighted, track_region_topk, "
            "or mean_topk_plus_track_region"
        )
    if scoring.variant not in scoring.variants:
        raise ValueError("scoring.variant must also be present in scoring.variants")
    if not 0 < scoring.topk_fraction <= 1:
        raise ValueError("scoring.topk_fraction must be in (0, 1]")
    if scoring.low_weight_alpha < 0:
        raise ValueError("scoring.low_weight_alpha must be non-negative")
    if scoring.low_weight_eps <= 0:
        raise ValueError("scoring.low_weight_eps must be positive")
    if scoring.beta < 0:
        raise ValueError("scoring.beta must be non-negative")
    if not 0 < scoring.motion_topk_fraction <= 1:
        raise ValueError("scoring.motion_topk_fraction must be in (0, 1]")
    if scoring.track_weight_alpha < 0:
        raise ValueError("scoring.track_weight_alpha must be non-negative")
    if scoring.track_speed_alpha < 0:
        raise ValueError("scoring.track_speed_alpha must be non-negative")
    if scoring.track_trajectory_alpha < 0:
        raise ValueError("scoring.track_trajectory_alpha must be non-negative")
    if scoring.track_weight_max < 1:
        raise ValueError("scoring.track_weight_max must be >= 1")
    if scoring.track_region_beta < 0:
        raise ValueError("scoring.track_region_beta must be non-negative")
    if any(beta < 0 for beta in scoring.sweep.betas):
        raise ValueError("scoring.sweep.betas must be non-negative")
    if any(not 0 < fraction <= 1 for fraction in scoring.sweep.topk_fractions):
        raise ValueError("scoring.sweep.topk_fractions must be in (0, 1]")
    if scoring.score_normalization.primary not in {
        SCORE_PRIMARY_RAW,
        SCORE_PRIMARY_VIDEO_CENTERED,
        SCORE_PRIMARY_ROLLING_CENTERED,
        SCORE_PRIMARY_SCENE_CALIBRATED,
    }:
        raise ValueError(
            "scoring.score_normalization.primary must be raw, video_centered, "
            "rolling_centered, or scene_calibrated"
        )
    if scoring.score_normalization.rolling_window <= 0:
        raise ValueError("scoring.score_normalization.rolling_window must be positive")
    if scoring.score_normalization.rolling_min_history < 0:
        raise ValueError("scoring.score_normalization.rolling_min_history must be non-negative")
    if any(window <= 0 for window in scoring.score_normalization.rolling_windows):
        raise ValueError("scoring.score_normalization.rolling_windows must be positive")
    if scoring.calibration.eps <= 0:
        raise ValueError("scoring.calibration.eps must be positive")
    if scoring.calibration.enabled and scoring.calibration.stats_path is None:
        raise ValueError("scoring.calibration.enabled requires scoring.calibration.stats_path")
    if scoring.decoded_mse.enabled:
        raise ValueError("scoring.decoded_mse.enabled=true is reserved for a later stage")


def parse_optional_int(value: Any) -> int | None:
    """Parse null or int config values."""

    if value is None:
        return None
    return int(value)


def parse_optional_float(value: Any) -> float | None:
    """Parse null or float config values."""

    if value is None:
        return None
    if isinstance(value, str) and not value:
        return None
    return float(value)


def parse_optional_str(value: Any) -> str | None:
    """Parse null or string config values."""

    if value is None:
        return None
    value = str(value)
    return value if value else None


def parse_optional_path(value: Any) -> Path | None:
    """Parse null or path-like config values."""

    parsed = parse_optional_str(value)
    return Path(parsed) if parsed else None


def parse_float_tuple(value: Any) -> tuple[float, ...]:
    """Parse a scalar or sequence into a float tuple."""

    if value is None:
        return ()
    if isinstance(value, (str, int, float)):
        return (float(value),)
    return tuple(float(item) for item in value)


def parse_int_tuple(value: Any) -> tuple[int, ...]:
    """Parse a scalar or sequence into an int tuple."""

    if value is None:
        return ()
    if isinstance(value, (str, int)):
        return (int(value),)
    return tuple(int(item) for item in value)


def load_feature_records(
    path: Path,
    *,
    normal_only: bool,
    limit_samples: int | None,
    require_track_features: bool = False,
    require_low_features: bool = False,
) -> list[dict[str, Any]]:
    """Load feature index records."""

    records = read_jsonl(path)
    if normal_only:
        records = [record for record in records if int(record.get("future_label", 0)) == 0]
    if limit_samples is not None:
        if limit_samples <= 0:
            raise ValueError("limit_samples must be positive")
        records = records[:limit_samples]
    if not records:
        raise ValueError(f"No feature records available from {path}")
    for record in records:
        if "high_feature_path" not in record or "z_path" not in record:
            raise ValueError("Feature index records must contain high_feature_path and z_path")
    has_low = ["low_feature_path" in record for record in records]
    if any(has_low) and not all(has_low):
        raise ValueError("Feature index records must consistently contain low_feature_path")
    if require_low_features and not all(has_low):
        missing = [
            str(record.get("sample_id", index))
            for index, record in enumerate(records)
            if "low_feature_path" not in record
        ]
        preview = ", ".join(missing[:5])
        raise ValueError(
            "condition.mode requires every feature record to contain low_feature_path; "
            f"missing samples: {preview}"
        )
    if require_track_features:
        missing = [
            str(record.get("sample_id", index))
            for index, record in enumerate(records)
            if "track_feature_path" not in record
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                "condition.use_track_grid=true requires every feature record to contain "
                f"track_feature_path; missing samples: {preview}"
            )
    return records


def split_train_val_records(
    records: list[dict[str, Any]],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into train and validation sets."""

    if val_fraction <= 0 or len(records) < 2:
        return records, []
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(records), generator=generator).tolist()
    val_count = max(1, int(round(len(records) * val_fraction)))
    val_indices = set(permutation[:val_count])
    train_records = [record for index, record in enumerate(records) if index not in val_indices]
    val_records = [record for index, record in enumerate(records) if index in val_indices]
    if not train_records:
        return records, []
    return train_records, val_records


def infer_z_dit_shape(
    sample: dict[str, Any],
    *,
    use_track_grid: bool = False,
    condition_mode: str = CONDITION_MODE_BASELINE,
    z_patch_size: int = 1,
) -> ZDiTShape:
    """Infer model input shapes from one dataset sample."""

    normalized_mode = normalize_condition_mode(condition_mode)
    requires_low = condition_mode_uses_appearance(normalized_mode)
    requires_track = bool(use_track_grid) or condition_mode_uses_track(normalized_mode)
    high = sample["high"]
    z = sample["z"]
    low = sample.get("low")
    track = sample.get("track")
    if high.ndim != 3:
        raise ValueError("Cached high embedding must be shaped [T, N, D]")
    if z.ndim != 4:
        raise ValueError("Cached Z must be shaped [T, C, H, W]")
    if low is None and requires_low:
        raise ValueError(f"condition.mode={normalized_mode} requires loaded low features")
    if low is not None and low.ndim != 4:
        raise ValueError("Cached low feature must be shaped [T, C, H, W]")
    track_grid = None
    if requires_track:
        if z_patch_size <= 0:
            raise ValueError("z_patch_size must be positive")
        if track is None:
            raise ValueError(
                f"condition.mode={normalized_mode} or condition.use_track_grid=true "
                "requires a loaded track feature payload"
            )
        track_grid = track.get("context_grid")
        if not isinstance(track_grid, torch.Tensor) or track_grid.ndim != 4:
            raise ValueError("Track context_grid must be shaped [T_ctx, C_track, H_grid, W_grid]")
        if track_grid.shape[1] != len(TRACK_CHANNELS):
            raise ValueError(
                "Track context_grid channel count must match TRACK_CHANNELS: "
                f"got {track_grid.shape[1]}, expected {len(TRACK_CHANNELS)}"
            )
        if z.shape[-2] % z_patch_size or z.shape[-1] % z_patch_size:
            raise ValueError("z_patch_size must divide cached Z height and width")
        expected_grid = (z.shape[-2] // z_patch_size, z.shape[-1] // z_patch_size)
        if tuple(track_grid.shape[-2:]) != expected_grid:
            raise ValueError(
                "Track context_grid spatial shape must match current Z patch grid: "
                f"got {tuple(track_grid.shape[-2:])}, expected {expected_grid}"
            )
    return ZDiTShape(
        high_frames=high.shape[0],
        high_tokens=high.shape[1],
        high_dim=high.shape[2],
        future_frames=z.shape[0],
        z_channels=z.shape[1],
        z_height=z.shape[2],
        z_width=z.shape[3],
        low_frames=low.shape[0] if low is not None else None,
        low_channels=low.shape[1] if low is not None else None,
        low_height=low.shape[2] if low is not None else None,
        low_width=low.shape[3] if low is not None else None,
        track_frames=track_grid.shape[0] if track_grid is not None else None,
        track_channels=track_grid.shape[1] if track_grid is not None else None,
        track_grid_h=track_grid.shape[2] if track_grid is not None else None,
        track_grid_w=track_grid.shape[3] if track_grid is not None else None,
    )


def z_dit_shape_from_dict(shape: dict[str, Any]) -> ZDiTShape:
    """Build a Z DiT shape object from checkpoint metadata."""

    allowed = set(ZDiTShape.__dataclass_fields__)
    return ZDiTShape(**{key: value for key, value in shape.items() if key in allowed})


def build_model(config: ModelConfig, shape: ZDiTShape) -> ConditionedZDiT:
    """Build a Z DiT from resolved config and inferred shapes."""

    return ConditionedZDiT(
        shape=shape,
        hidden_size=config.dit.hidden_size,
        num_layers=config.dit.num_layers,
        num_heads=config.dit.num_heads,
        mlp_ratio=config.dit.mlp_ratio,
        dropout=config.dit.dropout,
        high_max_tokens=config.high_adapter.max_tokens,
        high_use_pos_embedding=config.high_adapter.use_pos_embedding,
        high_token_reduction=config.high_adapter.token_reduction,
        high_foreground_ratio=config.high_adapter.foreground_ratio,
        z_patch_size=config.z_adapter.patch_size,
        z_use_temporal_pos=config.z_adapter.use_temporal_pos,
        z_use_spatial_pos=config.z_adapter.use_spatial_pos,
        low_patch_size=config.low_adapter.patch_size,
        low_use_temporal_pos=config.low_adapter.use_temporal_pos,
        low_use_spatial_pos=config.low_adapter.use_spatial_pos,
        condition_mode=config.condition.mode,
        use_track_grid=config.condition.use_track_grid,
        track_grid_gate_init=config.condition.track_grid_gate,
        gradient_checkpointing=config.dit.gradient_checkpointing,
    )


def build_flow_matching_sampler(config: FlowMatchingConfig) -> FlowMatchingSampler:
    """Build a flow-matching trainer/sampler from resolved config."""

    return FlowMatchingSampler(
        inference_steps=config.inference_steps,
        timestep_distribution=config.timestep_distribution,
        time_embedding_scale=config.time_embedding_scale,
        normalize_z=config.normalize_z,
        beta_alpha=config.beta_alpha,
        beta_beta=config.beta_beta,
        beta_s=config.beta_s,
    )


def torch_dtype(dtype: str) -> torch.dtype:
    """Return a torch dtype from a config string."""

    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def amp_is_enabled(config: TrainingConfig, device: torch.device) -> bool:
    """Return whether AMP should be enabled."""

    return config.amp and device.type == "cuda" and config.dtype in {"float16", "bfloat16"}


def make_loader_kwargs(
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
    prefetch_factor: int = 4,
) -> dict[str, Any]:
    """Return DataLoader kwargs shared by train/inference."""

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": feature_collate,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def move_feature_batch(
    batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Move high/Z/low tensors from a collated feature batch to device."""

    low = batch["low"]
    return (
        batch["high"].to(device, non_blocking=True),
        batch["z"].to(device, non_blocking=True),
        low.to(device, non_blocking=True) if low is not None else None,
    )


def move_track_context_grid(
    batch: dict[str, Any],
    device: torch.device,
    *,
    enabled: bool,
    z_patch_size: int,
) -> torch.Tensor | None:
    """Move optional C_track context grid to device and validate Z-grid alignment."""

    track = batch.get("track")
    if not enabled:
        return None
    if track is None:
        raise ValueError("condition.use_track_grid=true requires track features in every batch")
    context_grid = track["context_grid"]
    if context_grid.ndim != 5:
        raise ValueError("Track context_grid must be shaped [B, T_ctx, C_track, H_grid, W_grid]")
    if context_grid.shape[2] != len(TRACK_CHANNELS):
        raise ValueError(
            "Track context_grid channel count must match TRACK_CHANNELS: "
            f"got {context_grid.shape[2]}, expected {len(TRACK_CHANNELS)}"
        )
    z = batch["z"]
    if z.shape[-2] % z_patch_size or z.shape[-1] % z_patch_size:
        raise ValueError("z_patch_size must divide cached Z height and width")
    expected_grid = (z.shape[-2] // z_patch_size, z.shape[-1] // z_patch_size)
    if tuple(context_grid.shape[-2:]) != expected_grid:
        raise ValueError(
            "Track context_grid spatial shape must match current Z patch grid: "
            f"got {tuple(context_grid.shape[-2:])}, expected {expected_grid}"
        )
    return context_grid.to(device, non_blocking=True)


def apply_track_grid_ablation(
    context_grid: torch.Tensor,
    mode: str,
    seed: int,
) -> torch.Tensor:
    """Apply deterministic C_track ablation while preserving tensor shape."""

    normalized_mode = normalize_track_ablation_mode(mode)
    if normalized_mode == TRACK_ABLATION_REAL:
        return context_grid
    if normalized_mode == TRACK_ABLATION_ZERO:
        return torch.zeros_like(context_grid)
    if normalized_mode == TRACK_ABLATION_SHUFFLED:
        if context_grid.ndim < 1:
            raise ValueError("context_grid must have a batch dimension")
        batch = context_grid.shape[0]
        if batch <= 1:
            return context_grid.clone()
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        permutation = torch.randperm(batch, generator=generator, device="cpu")
        identity = torch.arange(batch, device="cpu")
        if torch.equal(permutation, identity):
            permutation = identity.roll(1)
        return context_grid.index_select(0, permutation.to(context_grid.device))
    raise ValueError(f"Unsupported track ablation mode: {mode}")


def apply_track_grid_channel_mask(
    context_grid: torch.Tensor,
    channels: str | list[str] | tuple[str, ...],
) -> torch.Tensor:
    """Zero all C_track channels except the selected channel mask."""

    selected_names = normalize_track_channel_names(channels)
    if tuple(selected_names) == tuple(TRACK_CHANNELS):
        return context_grid
    if context_grid.ndim != 5:
        raise ValueError("context_grid must be shaped [B, T_ctx, C_track, H_grid, W_grid]")
    if context_grid.shape[2] != len(TRACK_CHANNELS):
        raise ValueError(
            "context_grid channel count must match TRACK_CHANNELS: "
            f"got {context_grid.shape[2]}, expected {len(TRACK_CHANNELS)}"
        )
    selected = {TRACK_CHANNELS.index(name) for name in selected_names}
    output = torch.zeros_like(context_grid)
    for channel_index in selected:
        output[:, :, channel_index] = context_grid[:, :, channel_index]
    return output


def apply_track_grid_gate_override(model_or_adapter: Any, value: float | None) -> float | None:
    """Override a model C_track gate parameter at inference time."""

    target = getattr(model_or_adapter, "_orig_mod", model_or_adapter)
    gate = getattr(target, "track_grid_gate", None)
    if value is None:
        return read_track_grid_gate_value(target)
    if value < 0:
        raise ValueError("track_grid gate override must be non-negative")
    if gate is None:
        raise ValueError("track_grid gate override requested but model has no track_grid_gate")
    with torch.no_grad():
        gate.copy_(torch.as_tensor(float(value), dtype=gate.dtype, device=gate.device))
    return float(gate.detach().cpu())


def read_track_grid_gate_value(model_or_adapter: Any) -> float | None:
    """Return the current C_track gate value when present."""

    target = getattr(model_or_adapter, "_orig_mod", model_or_adapter)
    gate = getattr(target, "track_grid_gate", None)
    if gate is None:
        return None
    return float(gate.detach().cpu())


def normalize_track_ablation_mode(mode: str) -> str:
    """Normalize track ablation mode aliases."""

    key = str(mode).strip().lower()
    if key not in TRACK_ABLATION_ALIASES:
        raise ValueError("track_grid.ablation_mode must be real, zero, or shuffled")
    return TRACK_ABLATION_ALIASES[key]


def display_track_ablation_mode(mode: str) -> str:
    """Return the user-facing ablation mode name for reports."""

    normalized = normalize_track_ablation_mode(mode)
    return {
        TRACK_ABLATION_REAL: "real_track",
        TRACK_ABLATION_ZERO: "zero_track",
        TRACK_ABLATION_SHUFFLED: "shuffled_track",
    }[normalized]


def normalize_track_channel_names(channels: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Normalize a named C_track channel mask to channel names."""

    if isinstance(channels, str):
        key = channels.strip().lower()
        if key in TRACK_CHANNEL_MASKS:
            values = TRACK_CHANNEL_MASKS[key]
        else:
            values = tuple(item.strip().lower() for item in key.split(",") if item.strip())
    else:
        values = tuple(str(item).strip().lower() for item in channels)
    if not values:
        raise ValueError("track channel mask must select at least one channel")
    unknown = sorted(set(values) - set(TRACK_CHANNELS))
    if unknown:
        raise ValueError(f"unknown track channel names: {unknown}")
    return tuple(values)


def compute_z_stats(
    dataset: FeatureZDataset,
    *,
    batch_size: int = 16,
    num_workers: int = 0,
) -> dict[str, torch.Tensor]:
    """Compute channel-wise train Z mean/std for normalized scoring."""

    channel_sum: torch.Tensor | None = None
    channel_sumsq: torch.Tensor | None = None
    count = 0
    loader = DataLoader(
        dataset,
        **make_loader_kwargs(batch_size, num_workers, shuffle=False),
    )
    for batch in progress_bar(loader, desc="compute Z stats", unit="batch", leave=False):
        z = batch["z"]
        dims = (0, 1, 3, 4)
        current_sum = z.sum(dim=dims)
        current_sumsq = (z * z).sum(dim=dims)
        current_count = z.shape[0] * z.shape[1] * z.shape[3] * z.shape[4]
        channel_sum = current_sum if channel_sum is None else channel_sum + current_sum
        channel_sumsq = current_sumsq if channel_sumsq is None else channel_sumsq + current_sumsq
        count += current_count
    if channel_sum is None or channel_sumsq is None or count == 0:
        raise ValueError("Cannot compute Z stats from an empty dataset")
    mean = channel_sum / count
    variance = (channel_sumsq / count) - mean.square()
    std = variance.clamp_min(1e-12).sqrt()
    return {
        "mean": mean.reshape(1, 1, -1, 1, 1),
        "std": std.reshape(1, 1, -1, 1, 1),
    }


def per_frame_z_scores(
    z: torch.Tensor,
    z_hat: torch.Tensor,
    *,
    scoring: ScoringConfig,
    patch_size: int,
    z_stats: dict[str, torch.Tensor] | None,
    low: torch.Tensor | None = None,
    track_grid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return per-future-frame anomaly scores shaped ``[B, T]``."""

    variants = per_frame_z_score_variants(
        z,
        z_hat,
        scoring=scoring,
        patch_size=patch_size,
        z_stats=z_stats,
        low=low,
        track_grid=track_grid,
    )
    return variants[scoring.variant]


def per_frame_z_score_variants(
    z: torch.Tensor,
    z_hat: torch.Tensor,
    *,
    scoring: ScoringConfig,
    patch_size: int,
    z_stats: dict[str, torch.Tensor] | None,
    low: torch.Tensor | None = None,
    track_grid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return requested per-future-frame score variants shaped ``[B, T]``."""

    patch_errors = z_patch_errors(
        z,
        z_hat,
        scoring=scoring,
        patch_size=patch_size,
        z_stats=z_stats,
    )
    flat_errors = patch_errors.flatten(2)
    global_score = flat_errors.mean(dim=-1)
    patch_scores = {scoring.topk_fraction: topk_patch_score(flat_errors, scoring.topk_fraction)}
    for topk_fraction in scoring.sweep.topk_fractions:
        patch_scores.setdefault(topk_fraction, topk_patch_score(flat_errors, topk_fraction))
    patch_score = patch_scores[scoring.topk_fraction]
    scores: dict[str, torch.Tensor] = {}
    for variant in scoring.variants:
        if variant == SCORE_VARIANT_GLOBAL:
            scores[variant] = global_score
        elif variant == SCORE_VARIANT_PATCH:
            scores[variant] = patch_score
        elif variant == SCORE_VARIANT_LOW_WEIGHTED:
            scores[variant] = low_weighted_patch_score(
                patch_errors,
                low=low,
                alpha=scoring.low_weight_alpha,
                eps=scoring.low_weight_eps,
            )
        elif variant == SCORE_VARIANT_GLOBAL_PATCH:
            scores[variant] = global_score + scoring.beta * patch_score
        elif variant == SCORE_VARIANT_MOTION_PATCH:
            scores[variant] = motion_patch_score(
                patch_errors,
                low=low,
                topk_fraction=scoring.topk_fraction,
                motion_fraction=scoring.motion_topk_fraction,
            )
        elif variant == SCORE_VARIANT_MOTION_GLOBAL_PATCH:
            motion_score = motion_patch_score(
                patch_errors,
                low=low,
                topk_fraction=scoring.topk_fraction,
                motion_fraction=scoring.motion_topk_fraction,
            )
            scores[variant] = global_score + scoring.beta * motion_score
        elif variant == SCORE_VARIANT_TRACK_WEIGHTED:
            scores[variant] = track_weighted_patch_score(
                patch_errors,
                track_grid=track_grid,
                scoring=scoring,
            )
        elif variant == SCORE_VARIANT_TRACK_REGION_TOPK:
            scores[variant] = track_region_topk_score(
                patch_errors,
                track_grid=track_grid,
                topk_fraction=scoring.topk_fraction,
            )
        elif variant == SCORE_VARIANT_MEAN_TOPK_PLUS_TRACK_REGION:
            region_score, has_region = track_region_topk_score(
                patch_errors,
                track_grid=track_grid,
                topk_fraction=scoring.topk_fraction,
                return_has_region=True,
            )
            combined = (
                global_score + scoring.beta * patch_score + scoring.track_region_beta * region_score
            )
            scores[variant] = torch.where(has_region, combined, global_score)
        else:
            raise ValueError(f"Unsupported score variant: {variant}")
    for topk_fraction, current_patch_score in patch_scores.items():
        suffix = format_score_param(topk_fraction)
        scores[f"patch_k{suffix}"] = current_patch_score
        for beta in scoring.sweep.betas:
            beta_suffix = format_score_param(beta)
            scores[f"global_patch_b{beta_suffix}_k{suffix}"] = (
                global_score + beta * current_patch_score
            )
    return scores


def z_patch_errors(
    z: torch.Tensor,
    z_hat: torch.Tensor,
    *,
    scoring: ScoringConfig,
    patch_size: int,
    z_stats: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    """Return Z reconstruction errors on the DiT patch grid shaped ``[B, T, H, W]``."""

    if z.shape != z_hat.shape:
        raise ValueError("z and z_hat must have the same shape")
    if z.ndim != 5:
        raise ValueError("z and z_hat must be shaped [B, T, C, H, W]")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    error = z - z_hat
    if scoring.method == SCORE_NORMALIZED_Z_MSE:
        if z_stats is None:
            raise ValueError("normalized_z_mse requires z_stats")
        std = z_stats["std"].to(error.device).clamp_min(scoring.normalized_z_mse.eps)
        error = error / std
    batch, frames, channels, height, width = error.shape
    if height % patch_size or width % patch_size:
        raise ValueError("patch_size must divide Z height and width")
    patch_errors = error.square().reshape(batch * frames, channels, height, width)
    patch_errors = patch_errors.unfold(2, patch_size, patch_size).unfold(
        3,
        patch_size,
        patch_size,
    )
    patch_errors = patch_errors.mean(dim=(1, 4, 5))
    return patch_errors.reshape(batch, frames, height // patch_size, width // patch_size)


def topk_patch_score(flat_errors: torch.Tensor, topk_fraction: float) -> torch.Tensor:
    """Average the highest-error Z patches for each future frame."""

    num_patches = flat_errors.shape[-1]
    k = max(1, min(num_patches, math.ceil(num_patches * topk_fraction)))
    return flat_errors.topk(k, dim=-1).values.mean(dim=-1)


def format_score_param(value: float) -> str:
    """Format a float for stable score variant names."""

    return f"{value:g}".replace("-", "m").replace(".", "p")


def motion_patch_score(
    patch_errors: torch.Tensor,
    *,
    low: torch.Tensor | None,
    topk_fraction: float,
    motion_fraction: float,
) -> torch.Tensor:
    """Score high-error patches inside high-motion C_low regions."""

    if low is None:
        return topk_patch_score(patch_errors.flatten(2), topk_fraction)
    if low.ndim != 5 or low.shape[0] != patch_errors.shape[0]:
        raise ValueError(
            "low must be shaped [B, T, C, H, W] and match Z batch size "
            f"when motion scoring is enabled: got {tuple(low.shape)}"
        )

    batch, frames, z_h, z_w = patch_errors.shape
    motion = low.float().abs().mean(dim=2).mean(dim=1, keepdim=True)
    pooled_motion = torch.nn.functional.adaptive_avg_pool2d(
        motion,
        output_size=(z_h, z_w),
    ).flatten(1)
    flat_errors = patch_errors.flatten(2)
    num_patches = flat_errors.shape[-1]
    motion_k = max(1, min(num_patches, math.ceil(num_patches * motion_fraction)))
    error_k = max(1, min(motion_k, math.ceil(motion_k * topk_fraction)))
    indices = pooled_motion.topk(motion_k, dim=-1).indices
    indices = indices[:, None, :].expand(batch, frames, motion_k)
    selected_errors = flat_errors.gather(dim=-1, index=indices)
    return selected_errors.topk(error_k, dim=-1).values.mean(dim=-1)


def low_weighted_patch_score(
    patch_errors: torch.Tensor,
    *,
    low: torch.Tensor | None,
    alpha: float,
    eps: float,
) -> torch.Tensor:
    """Weight Z patch errors by past frame-difference low pooled to the Z grid."""

    if low is None or alpha == 0:
        return patch_errors.flatten(2).mean(dim=-1)

    batch, _, z_h, z_w = patch_errors.shape
    if low.ndim != 5 or low.shape[0] != batch:
        raise ValueError(
            "low must be shaped [B, T, C, H, W] and match Z batch size "
            f"when low-weighted scoring is enabled: got {tuple(low.shape)}"
        )
    low_magnitude = low.float().abs().mean(dim=2).mean(dim=1, keepdim=True)
    pooled_low = torch.nn.functional.adaptive_avg_pool2d(
        low_magnitude,
        output_size=(z_h, z_w),
    ).squeeze(1)
    low_scale = pooled_low.mean(dim=(1, 2), keepdim=True).clamp_min(eps)
    weights = 1.0 + alpha * (pooled_low / low_scale)
    weights = weights[:, None, :, :]
    return (patch_errors * weights).sum(dim=(2, 3)) / weights.sum(dim=(2, 3)).clamp_min(eps)


def track_weighted_patch_score(
    patch_errors: torch.Tensor,
    *,
    track_grid: torch.Tensor | None,
    scoring: ScoringConfig,
) -> torch.Tensor:
    """Weight Z patch errors by dense object/track grids."""

    flat_errors = patch_errors.flatten(2)
    global_score = flat_errors.mean(dim=-1)
    if track_grid is None:
        return global_score

    flat_grid = aligned_track_grid(track_grid, patch_errors)
    objectness = flat_grid[:, :, 0]
    speed = normalize_track_speed(flat_grid[:, :, 3], eps=scoring.low_weight_eps)
    trajectory = flat_grid[:, :, 4]
    weights = (
        1.0
        + scoring.track_weight_alpha * objectness
        + scoring.track_speed_alpha * speed
        + scoring.track_trajectory_alpha * trajectory
    )
    weights = weights.clamp(1.0, scoring.track_weight_max)
    return (flat_errors * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(
        scoring.low_weight_eps
    )


def track_region_topk_score(
    patch_errors: torch.Tensor,
    *,
    track_grid: torch.Tensor | None,
    topk_fraction: float,
    return_has_region: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Average top-k errors inside object or trajectory regions."""

    flat_errors = patch_errors.flatten(2)
    global_score = flat_errors.mean(dim=-1)
    if track_grid is None:
        has_region = torch.zeros_like(global_score, dtype=torch.bool)
        return (global_score, has_region) if return_has_region else global_score

    flat_grid = aligned_track_grid(track_grid, patch_errors)
    region = (flat_grid[:, :, 0] > 0) | (flat_grid[:, :, 4] > 0)
    scores = torch.empty_like(global_score)
    has_region = region.any(dim=-1)
    batch, frames, _ = flat_errors.shape
    for batch_index in range(batch):
        for frame_index in range(frames):
            mask = region[batch_index, frame_index]
            if not bool(mask.any()):
                scores[batch_index, frame_index] = global_score[batch_index, frame_index]
                continue
            values = flat_errors[batch_index, frame_index][mask]
            k = max(1, min(values.numel(), math.ceil(values.numel() * topk_fraction)))
            scores[batch_index, frame_index] = values.topk(k).values.mean()
    return (scores, has_region) if return_has_region else scores


def aligned_track_grid(track_grid: torch.Tensor, patch_errors: torch.Tensor) -> torch.Tensor:
    """Return track grid flattened row-major after patch-grid alignment checks."""

    if track_grid.ndim != 5:
        raise ValueError("track_grid must be shaped [B, T, C, H, W]")
    if track_grid.shape[:2] != patch_errors.shape[:2]:
        raise ValueError(
            "track_grid batch/frame shape must match patch errors: "
            f"got {tuple(track_grid.shape[:2])}, expected {tuple(patch_errors.shape[:2])}"
        )
    if track_grid.shape[2] != len(TRACK_CHANNELS):
        raise ValueError(
            f"track_grid channel count must be {len(TRACK_CHANNELS)} "
            f"{TRACK_CHANNELS}: got {track_grid.shape[2]}"
        )
    grid_patches = int(track_grid.shape[-2]) * int(track_grid.shape[-1])
    expected_patches = int(patch_errors.shape[-2]) * int(patch_errors.shape[-1])
    if grid_patches != expected_patches:
        raise ValueError(
            "track_grid patch count must match Z patch errors: "
            f"H_grid*W_grid={grid_patches}, expected {expected_patches}"
        )
    return track_grid.float().flatten(-2)


def normalize_track_speed(speed: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Normalize speed to [0, 1] using percentile-95 with a safe max fallback."""

    positive = speed[speed > 0].float()
    if positive.numel() == 0:
        return torch.zeros_like(speed, dtype=torch.float32)
    scale = torch.quantile(positive, 0.95).clamp_min(eps)
    max_value = positive.max().clamp_min(eps)
    scale = torch.where(scale > eps, scale, max_value)
    return (speed.float() / scale).clamp(0.0, 1.0)


def aggregate_frame_scores(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate future-frame scores into final frame-level scores."""

    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        video_id = record["video_id"]
        frames = record["future_frames"]
        labels = record["future_frame_labels"]
        scores = record["future_frame_scores"]
        score_variants = record.get("future_frame_score_variants", {})
        for future_offset, (frame_idx, label, score) in enumerate(
            zip(frames, labels, scores, strict=True)
        ):
            key = (video_id, int(frame_idx))
            bucket = grouped.setdefault(
                key,
                {
                    "video_id": video_id,
                    "scene_id": record.get("scene_id"),
                    "frame_idx": int(frame_idx),
                    "label": int(label),
                    "score_sum": 0.0,
                    "num_votes": 0,
                },
            )
            bucket["score_sum"] += float(score)
            bucket["num_votes"] += 1
            bucket["label"] = max(bucket["label"], int(label))
            bucket["scene_id"] = bucket["scene_id"] or record.get("scene_id")
            for variant, variant_scores in score_variants.items():
                bucket.setdefault("variant_score_sums", {})
                variant_sums = bucket["variant_score_sums"]
                variant_sums[variant] = variant_sums.get(variant, 0.0) + float(
                    variant_scores[future_offset]
                )

    frame_scores = []
    for _, bucket in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        score = bucket["score_sum"] / bucket["num_votes"]
        frame_record = {
            "video_id": bucket["video_id"],
            "scene_id": bucket["scene_id"],
            "frame_idx": bucket["frame_idx"],
            "score": score,
            "label": bucket["label"],
            "num_votes": bucket["num_votes"],
        }
        for variant, score_sum in bucket.get("variant_score_sums", {}).items():
            frame_record[f"{variant}_raw_score"] = score_sum / bucket["num_votes"]
        if "variant_score_sums" not in bucket:
            frame_record[f"{SCORE_VARIANT_GLOBAL}_raw_score"] = score
        frame_scores.append(frame_record)
    return frame_scores


def apply_score_normalization(
    frame_scores: list[dict[str, Any]],
    *,
    scoring: ScoringConfig,
) -> list[dict[str, Any]]:
    """Add centered score variants and set the primary ``score`` field."""

    if not frame_scores:
        return frame_scores

    raw_variant_keys = sorted(
        key for key in frame_scores[0] if key.endswith("_raw_score") and key != "raw_score"
    )
    if not raw_variant_keys:
        for record in frame_scores:
            record[f"{SCORE_VARIANT_GLOBAL}_raw_score"] = float(record["score"])
        raw_variant_keys = [f"{SCORE_VARIANT_GLOBAL}_raw_score"]

    primary_raw_key = f"{scoring.variant}_raw_score"
    if primary_raw_key not in frame_scores[0]:
        primary_raw_key = raw_variant_keys[0]

    for record in frame_scores:
        record["raw_score"] = float(record[primary_raw_key])
        record["video_centered_score"] = record["raw_score"]
        record["video_z_score"] = record["raw_score"]
        record["rolling_centered_score"] = record["raw_score"]
        for raw_key in raw_variant_keys:
            prefix = raw_key.removesuffix("_raw_score")
            raw_score = float(record[raw_key])
            record[f"{prefix}_video_centered_score"] = raw_score
            record[f"{prefix}_video_z_score"] = raw_score
            record[f"{prefix}_rolling_centered_score"] = raw_score

    if not scoring.score_normalization.enabled:
        for record in frame_scores:
            record["score"] = float(record[primary_raw_key])
        add_track_score_aliases(frame_scores)
        return frame_scores

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in frame_scores:
        grouped.setdefault(str(record["video_id"]), []).append(record)

    for records in grouped.values():
        records.sort(key=lambda item: int(item["frame_idx"]))
        for raw_key in raw_variant_keys:
            prefix = raw_key.removesuffix("_raw_score")
            raw_scores = [float(record[raw_key]) for record in records]
            video_median = float(median(raw_scores))
            video_mean = sum(raw_scores) / len(raw_scores)
            if len(raw_scores) > 1:
                variance = sum((score - video_mean) ** 2 for score in raw_scores) / (
                    len(raw_scores) - 1
                )
                video_std = math.sqrt(max(variance, 1.0e-12))
            else:
                video_std = 1.0

            history: list[float] = []
            for record in records:
                raw_score = float(record[raw_key])
                record[f"{prefix}_video_centered_score"] = raw_score - video_median
                record[f"{prefix}_video_z_score"] = (raw_score - video_mean) / video_std
                if len(history) >= scoring.score_normalization.rolling_min_history:
                    window = history[-scoring.score_normalization.rolling_window :]
                    rolling_baseline = float(median(window))
                elif history:
                    rolling_baseline = float(median(history))
                else:
                    rolling_baseline = raw_score
                record[f"{prefix}_rolling_centered_score"] = raw_score - rolling_baseline
                for rolling_window in scoring.score_normalization.rolling_windows:
                    if len(history) >= scoring.score_normalization.rolling_min_history:
                        sweep_window = history[-rolling_window:]
                        sweep_baseline = float(median(sweep_window))
                    elif history:
                        sweep_baseline = float(median(history))
                    else:
                        sweep_baseline = raw_score
                    record[f"{prefix}_rolling_w{rolling_window}_centered_score"] = (
                        raw_score - sweep_baseline
                    )
                history.append(raw_score)

    for record in frame_scores:
        record["raw_score"] = float(record[primary_raw_key])
        primary_prefix = primary_raw_key.removesuffix("_raw_score")
        record["video_centered_score"] = float(record[f"{primary_prefix}_video_centered_score"])
        record["video_z_score"] = float(record[f"{primary_prefix}_video_z_score"])
        record["rolling_centered_score"] = float(record[f"{primary_prefix}_rolling_centered_score"])

    primary_key = {
        SCORE_PRIMARY_RAW: primary_raw_key,
        SCORE_PRIMARY_VIDEO_CENTERED: f"{primary_raw_key.removesuffix('_raw_score')}_video_centered_score",
        SCORE_PRIMARY_ROLLING_CENTERED: f"{primary_raw_key.removesuffix('_raw_score')}_rolling_centered_score",
        SCORE_PRIMARY_SCENE_CALIBRATED: primary_raw_key,
    }[scoring.score_normalization.primary]
    for record in frame_scores:
        record["score"] = float(record[primary_key])
    add_track_score_aliases(frame_scores)
    return frame_scores


def add_track_score_aliases(frame_scores: list[dict[str, Any]]) -> None:
    """Add raw-score aliases only when track-aware score variants are present."""

    if not frame_scores or f"{SCORE_VARIANT_TRACK_WEIGHTED}_raw_score" not in frame_scores[0]:
        return
    aliases = {
        "global_score": f"{SCORE_VARIANT_GLOBAL}_raw_score",
        "track_weighted_score": f"{SCORE_VARIANT_TRACK_WEIGHTED}_raw_score",
        "track_region_topk_score": f"{SCORE_VARIANT_TRACK_REGION_TOPK}_raw_score",
    }
    for record in frame_scores:
        for alias, source in aliases.items():
            if source in record:
                record[alias] = float(record[source])


def fit_score_calibration(
    frame_scores: list[dict[str, Any]],
    *,
    scoring: ScoringConfig,
) -> dict[str, Any]:
    """Fit group-wise median/MAD calibration stats from normal frame scores."""

    raw_keys = raw_score_keys(frame_scores)
    group_key = scoring.calibration.group_key
    stats: dict[str, Any] = {
        "group_key": group_key,
        "eps": scoring.calibration.eps,
        "scores": {},
    }
    normal_records = [record for record in frame_scores if int(record.get("label", 0)) == 0]
    if not normal_records:
        normal_records = frame_scores

    for raw_key in raw_keys:
        prefix = raw_key.removesuffix("_raw_score")
        grouped: dict[str, list[float]] = {"__global__": []}
        for record in normal_records:
            value = float(record[raw_key])
            group = str(record.get(group_key) or "__global__")
            grouped.setdefault(group, []).append(value)
            grouped["__global__"].append(value)
        stats["scores"][prefix] = {
            group: robust_location_scale(values, eps=scoring.calibration.eps)
            for group, values in grouped.items()
            if values
        }
    return stats


def apply_score_calibration(
    frame_scores: list[dict[str, Any]],
    *,
    scoring: ScoringConfig,
    calibration_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    """Add scene/video calibrated score variants using fitted normal stats."""

    if not frame_scores:
        return frame_scores
    raw_keys = raw_score_keys(frame_scores)
    group_key = str(calibration_stats.get("group_key", scoring.calibration.group_key))
    all_stats = calibration_stats.get("scores", {})
    primary_prefix = scoring.variant
    primary_key = f"{primary_prefix}_scene_calibrated_score"

    for record in frame_scores:
        group = str(record.get(group_key) or "__global__")
        for raw_key in raw_keys:
            prefix = raw_key.removesuffix("_raw_score")
            score_stats = all_stats.get(prefix, {})
            group_stats = score_stats.get(group) or score_stats.get("__global__")
            if not group_stats:
                continue
            median_value = float(group_stats["median"])
            scale = max(float(group_stats["mad"]), scoring.calibration.eps)
            calibrated = (float(record[raw_key]) - median_value) / scale
            record[f"{prefix}_scene_calibrated_score"] = calibrated
        if primary_key in record:
            record["scene_calibrated_score"] = float(record[primary_key])
            if scoring.score_normalization.primary == SCORE_PRIMARY_SCENE_CALIBRATED:
                record["score"] = float(record[primary_key])
    return frame_scores


def raw_score_keys(frame_scores: list[dict[str, Any]]) -> list[str]:
    """Return raw score keys available in frame records."""

    if not frame_scores:
        return []
    return sorted(
        key
        for key, value in frame_scores[0].items()
        if key.endswith("_raw_score") and isinstance(value, (int, float))
    )


def robust_location_scale(values: list[float], *, eps: float) -> dict[str, float]:
    """Return robust median/MAD statistics."""

    median_value = float(median(values))
    deviations = [abs(value - median_value) for value in values]
    mad = max(float(median(deviations)), eps)
    return {
        "count": float(len(values)),
        "median": median_value,
        "mad": mad,
    }


def compute_binary_metrics(
    frame_scores: list[dict[str, Any]],
    *,
    score_key: str = "score",
) -> dict[str, float | None | int]:
    """Compute frame-level ROC-AUC and average precision when labels allow it."""

    available_records = [record for record in frame_scores if score_key in record]
    labels = [int(record["label"]) for record in available_records]
    scores = [float(record[score_key]) for record in available_records]
    metrics: dict[str, float | None | int] = {
        "num_frames": len(available_records),
        "num_anomaly_frames": int(sum(labels)),
        "roc_auc": None,
        "average_precision": None,
    }
    if len(set(labels)) < 2:
        return metrics

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        metrics["metrics_error"] = "scikit-learn is not installed"
        return metrics

    metrics["roc_auc"] = float(roc_auc_score(labels, scores))
    metrics["average_precision"] = float(average_precision_score(labels, scores))
    return metrics


def compute_score_metrics(frame_scores: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute metrics for raw and centered score variants."""

    if not frame_scores:
        return {}

    base_score_keys = [
        "score",
        "raw_score",
        "video_centered_score",
        "video_z_score",
        "rolling_centered_score",
    ]
    extra_score_keys = sorted(
        key
        for key, value in frame_scores[0].items()
        if key.endswith("_score") and key not in base_score_keys and isinstance(value, (int, float))
    )
    score_keys = base_score_keys + extra_score_keys
    return {
        score_key: compute_binary_metrics(frame_scores, score_key=score_key)
        for score_key in score_keys
        if score_key in frame_scores[0]
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write a YAML payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL records."""

    write_jsonl(path, records)


def config_to_dict(config: ZDiTPipelineConfig) -> dict[str, Any]:
    """Return a JSON/YAML serializable resolved config."""

    resolved = asdict(config)
    resolved.pop("raw", None)
    return stringify_paths(resolved)


def stringify_paths(value: Any) -> Any:
    """Recursively convert paths to strings."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [stringify_paths(item) for item in value]
    return value
