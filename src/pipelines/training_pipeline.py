"""Training pipeline for C_high/C_low-conditioned Z flow DiT."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models import ConditionedZDiT, FlowMatchingSampler, ZDiTShape
from src.pipelines.z_dit_pipeline import (
    FeatureZDataset,
    ZDiTPipelineConfig,
    amp_is_enabled,
    build_flow_matching_sampler,
    build_model,
    compute_z_stats,
    condition_uses_appearance_features,
    condition_uses_legacy_track_gate,
    condition_uses_patch_condition,
    condition_uses_track_features,
    config_to_dict,
    infer_z_dit_shape,
    load_feature_records,
    load_pipeline_config,
    make_loader_kwargs,
    move_feature_batch,
    move_track_context_grid,
    split_train_val_records,
    torch_dtype,
    write_json,
    write_yaml,
    z_dit_shape_from_dict,
)
from src.utils import cleanup_memory, progress_bar, progress_write


def train(
    config_path: Path,
    *,
    run_id: str | None = None,
    init_checkpoint_path: Path | None = None,
    max_steps: int | None = None,
    limit_samples: int | None = None,
    overwrite: bool | None = None,
) -> Path:
    """Run Z DiT training and return the run directory."""

    config = load_pipeline_config(config_path)
    if max_steps is not None:
        config = replace(config, training=replace(config.training, max_steps=max_steps))
    active_run_id = run_id or config.training.run_id or default_run_id()
    run_dir = config.training.output_root / active_run_id
    prepare_run_dir(
        run_dir,
        overwrite=config.training.overwrite if overwrite is None else overwrite,
    )

    set_seed(config.training.seed)
    requires_track_features = condition_uses_track_features(config.model.condition)
    requires_low_features = condition_uses_appearance_features(config.model.condition)
    records = load_feature_records(
        config.training.feature_index,
        normal_only=True,
        limit_samples=limit_samples,
        require_track_features=requires_track_features,
        require_low_features=requires_low_features,
    )
    train_records, val_records = split_train_val_records(
        records,
        val_fraction=config.training.val_fraction,
        seed=config.training.seed,
    )
    train_dataset = FeatureZDataset(
        train_records,
        require_track_features=requires_track_features,
        require_low_features=requires_low_features,
    )
    val_dataset = (
        FeatureZDataset(
            val_records,
            require_track_features=requires_track_features,
            require_low_features=requires_low_features,
        )
        if val_records
        else None
    )
    shape = infer_z_dit_shape(
        train_dataset[0],
        use_track_grid=condition_uses_legacy_track_gate(config.model.condition),
        condition_mode=config.model.condition.mode,
        z_patch_size=config.model.z_adapter.patch_size,
    )

    configure_torch_backend(config.optimization)
    device = get_device()
    model = build_model(config.model, shape).to(device)
    active_init_checkpoint_path = init_checkpoint_path or config.training.init_checkpoint_path
    if active_init_checkpoint_path is not None:
        load_init_checkpoint_weights(model, shape, active_init_checkpoint_path, device=device)
    log_track_condition(model, shape, config)
    model = maybe_compile_model(model, enabled=config.training.compile)
    flow = build_flow_matching_sampler(config.flow_matching)
    optimizer = build_optimizer(model, config, device)

    z_stats = compute_z_stats(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    torch.save(z_stats, run_dir / "z_stats.pt")
    write_yaml(run_dir / "config_resolved.yaml", config_to_dict(config))
    write_json(
        run_dir / "run_metadata.json",
        {
            "run_id": active_run_id,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset) if val_dataset is not None else 0,
            "shape": asdict(shape),
            "objective": "conditional_flow_matching",
            "track_condition": track_condition_metadata(model, shape, config),
            "init_checkpoint_path": str(active_init_checkpoint_path)
            if active_init_checkpoint_path is not None
            else None,
        },
    )

    train_loader = DataLoader(
        train_dataset,
        **make_loader_kwargs(
            config.training.batch_size,
            config.training.num_workers,
            shuffle=True,
            prefetch_factor=config.optimization.prefetch_factor,
        ),
    )
    val_loader = (
        DataLoader(
            val_dataset,
            **make_loader_kwargs(
                config.training.batch_size,
                config.training.num_workers,
                shuffle=False,
                prefetch_factor=config.optimization.prefetch_factor,
            ),
        )
        if val_dataset is not None
        else None
    )

    state = TrainingState(step=0, best_loss=float("inf"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_is_enabled(config.training, device))
    metrics_path = run_dir / "metrics.jsonl"
    train_iter = iter(train_loader)
    progress = progress_bar(total=config.training.max_steps, desc="train Z DiT", unit="step")
    try:
        while state.step < config.training.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            state.step += 1
            losses = train_step(
                model,
                flow,
                optimizer,
                scaler,
                batch,
                config,
                z_stats,
                device,
            )
            progress.update(1)
            should_log = state.step % config.training.log_every_steps == 0 or state.step == 1
            if should_log:
                progress.set_postfix(
                    train_loss=f"{losses['loss']:.6f}",
                    global_loss=f"{losses['loss_global']:.6f}",
                    patch_loss=f"{losses['loss_patch']:.6f}",
                    refresh=False,
                )
                append_metric(
                    metrics_path,
                    {
                        "step": state.step,
                        "train_loss": losses["loss"],
                        "train_loss_global": losses["loss_global"],
                        "train_loss_patch": losses["loss_patch"],
                        "train_loss_patch_weighted": losses["loss_patch_weighted"],
                    },
                )

            should_save = (
                state.step % config.training.save_every_steps == 0
                or state.step == config.training.max_steps
            )
            if should_save:
                val_loss = (
                    evaluate_loss(model, flow, val_loader, config, z_stats, device)
                    if val_loader
                    else None
                )
                selected_loss = val_loss if val_loss is not None else losses["loss"]
                append_metric(
                    metrics_path,
                    {
                        "step": state.step,
                        "train_loss": losses["loss"],
                        "train_loss_global": losses["loss_global"],
                        "train_loss_patch": losses["loss_patch"],
                        "train_loss_patch_weighted": losses["loss_patch_weighted"],
                        "val_loss": val_loss,
                        "selected_loss": selected_loss,
                    },
                )
                progress.set_postfix(
                    train_loss=f"{losses['loss']:.6f}",
                    selected_loss=f"{selected_loss:.6f}",
                    refresh=False,
                )
                is_best = selected_loss < state.best_loss
                if is_best:
                    state.best_loss = selected_loss
                save_checkpoint(
                    run_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    shape=shape,
                    config=config,
                    z_stats=z_stats,
                    step=state.step,
                    best_loss=state.best_loss,
                )
                if is_best:
                    save_checkpoint(
                        run_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        shape=shape,
                        config=config,
                        z_stats=z_stats,
                        step=state.step,
                        best_loss=state.best_loss,
                    )
                    progress_write(f"saved best checkpoint -> {run_dir / 'best.pt'}")
    finally:
        progress.close()
        cleanup_memory(cuda=device.type == "cuda")

    write_json(
        run_dir / "metrics_summary.json",
        {
            "run_id": active_run_id,
            "best_loss": state.best_loss,
            "steps": state.step,
            "checkpoint": str(run_dir / "best.pt"),
            "init_checkpoint_path": str(active_init_checkpoint_path)
            if active_init_checkpoint_path is not None
            else None,
        },
    )
    return run_dir


def load_init_checkpoint_weights(
    model: nn.Module,
    shape: ZDiTShape,
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> None:
    """Initialize model weights from a compatible training checkpoint."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing init checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Init checkpoint is missing model_state_dict: {checkpoint_path}")
    if "shape" not in checkpoint:
        raise ValueError(f"Init checkpoint is missing shape metadata: {checkpoint_path}")

    checkpoint_shape = z_dit_shape_from_dict(checkpoint["shape"])
    allow_track_extension = (
        checkpoint_shape != shape
        and base_shape_dict(checkpoint_shape) == base_shape_dict(shape)
        and not shape_has_track_grid(checkpoint_shape)
        and shape_has_track_grid(shape)
    )
    patch_condition_adapter = getattr(model, "patch_condition_adapter", None)
    allow_patch_condition_init = patch_condition_adapter is not None and (
        checkpoint_shape == shape or base_shape_dict(checkpoint_shape) == base_shape_dict(shape)
    )
    allow_relaxed_init = allow_track_extension or allow_patch_condition_init
    if checkpoint_shape != shape and not allow_relaxed_init:
        raise ValueError(
            "Init checkpoint shape does not match current feature shape. "
            f"checkpoint={asdict(checkpoint_shape)} current={asdict(shape)}"
        )

    try:
        incompatible = model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=not allow_relaxed_init,
        )
    except RuntimeError as error:
        raise RuntimeError(f"Failed to load init checkpoint weights: {checkpoint_path}") from error
    if allow_relaxed_init:
        missing = sorted(incompatible.missing_keys)
        unexpected = sorted(incompatible.unexpected_keys)
        progress_write(
            "initialized model weights from checkpoint with strict=False; "
            f"new/missing parameters={missing}; unexpected={unexpected}"
        )
        return
    progress_write(f"initialized model weights from {checkpoint_path}")


def base_shape_dict(shape: ZDiTShape) -> dict[str, Any]:
    """Return shape metadata excluding optional C_track fields."""

    value = asdict(shape)
    for key in ("track_frames", "track_channels", "track_grid_h", "track_grid_w"):
        value.pop(key, None)
    return value


def shape_has_track_grid(shape: ZDiTShape) -> bool:
    """Return whether shape metadata includes C_track grid dimensions."""

    return all(
        value is not None
        for value in (
            shape.track_frames,
            shape.track_channels,
            shape.track_grid_h,
            shape.track_grid_w,
        )
    )


def track_condition_metadata(
    model: nn.Module,
    shape: ZDiTShape,
    config: ZDiTPipelineConfig,
) -> dict[str, Any]:
    """Return serializable condition metadata for logs/checkpoints."""

    unwrapped = unwrap_model(model)
    condition = config.model.condition
    enabled = condition_uses_track_features(condition)
    patch_condition = condition_uses_patch_condition(condition)
    gate = getattr(unwrapped, "track_grid_gate", None)
    token_shape = None
    if patch_condition:
        adapter = getattr(unwrapped, "patch_condition_adapter", None)
        if adapter is not None:
            token_shape = [
                1,
                int(adapter.condition_frames) * int(adapter.grid_h) * int(adapter.grid_w),
                int(config.model.dit.hidden_size),
            ]
    elif enabled and shape_has_track_grid(shape):
        token_shape = [
            1,
            int(shape.track_frames) * int(shape.track_grid_h) * int(shape.track_grid_w),
            int(config.model.dit.hidden_size),
        ]
    return {
        "enabled": enabled,
        "condition_mode": condition.mode,
        "patch_condition": patch_condition,
        "legacy_track_gate": condition_uses_legacy_track_gate(condition),
        "track_grid_gate": float(gate.detach().cpu()) if gate is not None else None,
        "context_token_shape": token_shape,
    }


def log_track_condition(
    model: nn.Module,
    shape: ZDiTShape,
    config: ZDiTPipelineConfig,
) -> None:
    """Log condition state once at startup."""

    metadata = track_condition_metadata(model, shape, config)
    progress_write(
        f"condition mode: {metadata['condition_mode']}; track grid enabled: {metadata['enabled']}"
    )
    if metadata["context_token_shape"] is not None:
        progress_write(f"condition token shape: {metadata['context_token_shape']}")
    if metadata["track_grid_gate"] is not None:
        progress_write(f"track_grid_gate={metadata['track_grid_gate']:.6f}")


def prepare_run_dir(run_dir: Path, *, overwrite: bool) -> None:
    """Create a clean run directory or fail before mixing stale checkpoints."""

    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Training run directory already contains files: {run_dir}. "
                "Use a new --run-id, remove the directory, or rerun train.py with --overwrite."
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


class TrainingState:
    """Mutable training loop state."""

    def __init__(self, *, step: int, best_loss: float) -> None:
        self.step = step
        self.best_loss = best_loss


def train_step(
    model: nn.Module,
    flow: FlowMatchingSampler,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: dict[str, Any],
    config: ZDiTPipelineConfig,
    z_stats: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    """Run one optimization step."""

    model.train()
    try:
        high, z, low = move_feature_batch(batch, device)
        track_grid = move_track_context_grid(
            batch,
            device,
            enabled=condition_uses_track_features(config.model.condition),
            z_patch_size=config.model.z_adapter.patch_size,
        )
        z_at_time, time_values, target_velocity, _ = flow.prepare_training_pair(
            z,
            z_stats=z_stats,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch_dtype(config.training.dtype),
            enabled=amp_is_enabled(config.training, device),
        ):
            prediction = model(z_at_time, time_values, high, low, track_grid)
            loss, loss_global, loss_patch = compute_velocity_loss(
                prediction,
                target_velocity,
                patch_size=config.model.z_adapter.patch_size,
                patch_alpha=config.loss.alpha,
                topk_fraction=config.loss.topk_fraction,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if config.training.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    except torch.OutOfMemoryError:
        optimizer.zero_grad(set_to_none=True)
        cleanup_memory(cuda=device.type == "cuda")
        raise
    return {
        "loss": float(loss.detach().cpu()),
        "loss_global": float(loss_global.detach().cpu()),
        "loss_patch": float(loss_patch.detach().cpu()),
        "loss_patch_weighted": float((config.loss.alpha * loss_patch).detach().cpu()),
    }


def build_optimizer(
    model: nn.Module,
    config: ZDiTPipelineConfig,
    device: torch.device,
) -> torch.optim.Optimizer:
    """Build AdamW, using CUDA fused kernels when configured and available."""

    kwargs: dict[str, Any] = {
        "lr": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
    }
    fused_mode = config.optimization.fused_adamw
    should_try_fused = device.type == "cuda" and fused_mode in {"auto", "true"}
    if should_try_fused:
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except (RuntimeError, TypeError) as exc:
            if fused_mode == "true":
                raise RuntimeError("optimization.fused_adamw=true failed") from exc
            progress_write(f"fused AdamW unavailable; falling back to standard AdamW: {exc}")
    return torch.optim.AdamW(model.parameters(), **kwargs)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    flow: FlowMatchingSampler,
    loader: DataLoader | None,
    config: ZDiTPipelineConfig,
    z_stats: dict[str, torch.Tensor],
    device: torch.device,
) -> float | None:
    """Evaluate flow-matching velocity MSE on a validation loader."""

    if loader is None:
        return None
    model.eval()
    losses = []
    try:
        for batch in progress_bar(loader, desc="validate Z DiT", unit="batch", leave=False):
            high, z, low = move_feature_batch(batch, device)
            track_grid = move_track_context_grid(
                batch,
                device,
                enabled=condition_uses_track_features(config.model.condition),
                z_patch_size=config.model.z_adapter.patch_size,
            )
            z_at_time, time_values, target_velocity, _ = flow.prepare_training_pair(
                z,
                z_stats=z_stats,
            )
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch_dtype(config.training.dtype),
                enabled=amp_is_enabled(config.training, device),
            ):
                prediction = model(z_at_time, time_values, high, low, track_grid)
                loss, _, _ = compute_velocity_loss(
                    prediction,
                    target_velocity,
                    patch_size=config.model.z_adapter.patch_size,
                    patch_alpha=config.loss.alpha,
                    topk_fraction=config.loss.topk_fraction,
                )
            losses.append(float(loss.cpu()))
    finally:
        cleanup_memory(cuda=device.type == "cuda")
    return sum(losses) / len(losses) if losses else None


def compute_velocity_loss(
    prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    *,
    patch_size: int,
    patch_alpha: float,
    topk_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, global, and top-k patch velocity losses."""

    prediction = prediction.float()
    target_velocity = target_velocity.float()
    loss_global = nn.functional.mse_loss(prediction, target_velocity)
    loss_patch = topk_patch_mse(
        prediction,
        target_velocity,
        patch_size=patch_size,
        topk_fraction=topk_fraction,
    )
    loss = loss_global + patch_alpha * loss_patch
    return loss, loss_global, loss_patch


def topk_patch_mse(
    prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    *,
    patch_size: int,
    topk_fraction: float,
) -> torch.Tensor:
    """Return mean top-k per-patch velocity MSE over the Z patch grid."""

    if prediction.shape != target_velocity.shape:
        raise ValueError("prediction and target_velocity must have the same shape")
    if prediction.ndim != 5:
        raise ValueError("prediction and target_velocity must be shaped [B, T, C, H, W]")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if not 0 < topk_fraction <= 1:
        raise ValueError("topk_fraction must be in (0, 1]")

    batch, frames, channels, height, width = prediction.shape
    if height % patch_size or width % patch_size:
        raise ValueError("patch_size must divide prediction height and width")

    error = (prediction - target_velocity).square().reshape(batch * frames, channels, height, width)
    patches = error.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patch_errors = patches.mean(dim=(1, 4, 5)).reshape(batch, frames, -1)
    num_patches = patch_errors.shape[-1]
    k = max(1, min(num_patches, math.ceil(num_patches * topk_fraction)))
    return patch_errors.topk(k, dim=-1).values.mean()


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    shape: ZDiTShape,
    config: ZDiTPipelineConfig,
    z_stats: dict[str, torch.Tensor],
    step: int,
    best_loss: float,
) -> None:
    """Save a training checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "shape": asdict(shape),
            "config": config_to_dict(config),
            "z_stats": z_stats,
            "step": step,
            "best_loss": best_loss,
        },
        path,
    )


def load_checkpoint_for_inference(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[ConditionedZDiT, dict[str, torch.Tensor], dict[str, Any]]:
    """Load a checkpoint and return model, Z stats, and checkpoint config."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint["config"]
    if "flow_matching" not in checkpoint_config:
        raise ValueError(
            "Checkpoint was not trained with the flow_matching pipeline. "
            "Retrain the Z DiT after the flow-matching migration."
        )
    config = load_config_from_resolved_dict(checkpoint_config)
    shape = z_dit_shape_from_dict(checkpoint["shape"])
    model = build_model(config.model, shape).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    z_stats = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in checkpoint["z_stats"].items()
    }
    return model, z_stats, checkpoint_config


def maybe_compile_model(model: ConditionedZDiT, *, enabled: bool) -> nn.Module:
    """Optionally compile a model with torch.compile."""

    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("training.compile/inference.compile requires torch.compile")
    return torch.compile(model)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the original module from a compiled wrapper when present."""

    return getattr(model, "_orig_mod", model)


def load_config_from_resolved_dict(config: dict[str, Any]) -> ZDiTPipelineConfig:
    """Reconstruct a pipeline config from a resolved checkpoint dict."""

    required_sections = {
        "model",
        "flow_matching",
        "loss",
        "training",
        "inference",
        "scoring",
    }
    missing_sections = sorted(required_sections - set(config))
    if missing_sections:
        raise ValueError(
            "Checkpoint config is missing required sections "
            f"{missing_sections}. This checkpoint is stale or incompatible with the current "
            "Z DiT pipeline. Retrain with a new --run-id or rerun train.py with --overwrite."
        )

    temp_path = Path("__resolved_config_should_not_be_read__.yaml")
    raw = {
        "dataset": {"name": config.get("dataset_name", "shanghaitech")},
        "model": config["model"],
        "flow_matching": config["flow_matching"],
        "loss": config["loss"],
        "optimization": config.get("optimization", {}),
        "training": config["training"],
        "inference": config["inference"],
        "scoring": config["scoring"],
    }
    resolved = load_pipeline_config_from_dict(raw)
    return ZDiTPipelineConfig(
        dataset_name=raw["dataset"]["name"],
        model=resolved.model,
        flow_matching=resolved.flow_matching,
        loss=resolved.loss,
        optimization=resolved.optimization,
        training=resolved.training,
        inference=resolved.inference,
        scoring=resolved.scoring,
        raw={"source": str(temp_path)},
    )


def load_pipeline_config_from_dict(raw: dict[str, Any]) -> ZDiTPipelineConfig:
    """Load config from an in-memory dictionary."""

    from src.pipelines.z_dit_pipeline import (
        load_flow_matching_config,
        load_inference_config,
        load_loss_config,
        load_model_config,
        load_optimization_config,
        load_scoring_config,
        load_training_config,
        validate_pipeline_config,
    )

    model = load_model_config(raw.get("model", {}))
    flow_matching = load_flow_matching_config(raw.get("flow_matching", {}))
    loss = load_loss_config(raw.get("loss", {}))
    optimization = load_optimization_config(raw.get("optimization", {}))
    training = load_training_config(raw.get("training", {}))
    inference = load_inference_config(raw.get("inference", {}))
    scoring = load_scoring_config(raw.get("scoring", {}))
    validate_pipeline_config(model, flow_matching, loss, optimization, training, inference, scoring)
    return ZDiTPipelineConfig(
        dataset_name=str(raw.get("dataset", {}).get("name", "shanghaitech")),
        model=model,
        flow_matching=flow_matching,
        loss=loss,
        optimization=optimization,
        training=training,
        inference=inference,
        scoring=scoring,
        raw=raw,
    )


def append_metric(path: Path, metric: dict[str, Any]) -> None:
    """Append one JSONL metric row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metric, sort_keys=True) + "\n")


def default_run_id() -> str:
    """Return a timestamp run id."""

    return datetime.now(timezone.utc).strftime("z_dit_%Y%m%d_%H%M%S")


def get_device() -> torch.device:
    """Return the active torch device."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed torch random state."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_backend(config: Any) -> None:
    """Apply configured torch backend optimizations when available."""

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(config.matmul_precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(config.enable_tf32)
        torch.backends.cudnn.allow_tf32 = bool(config.enable_tf32)
        torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark)
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(bool(config.enable_flash_sdp))
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
