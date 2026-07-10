"""Training pipeline for memory-guided fixed-grid RGB patch DiT."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.models import FlowMatcher, RGBPatchDiT
from src.pipelines.rgb_patch_pipeline import (
    RGBPatchDataset,
    RGBPatchModelConfig,
    RGBPatchPipelineConfig,
    RGBPatchShape,
    config_to_dict,
    infer_patch_shape,
    load_pipeline_config,
    load_sample_records,
    make_loader,
    split_train_val_records,
)
from src.utils import cleanup_memory, progress_bar, progress_write


def train(
    config_path: Path,
    *,
    run_id: str | None = None,
    max_steps: int | None = None,
    limit_samples: int | None = None,
    overwrite: bool | None = None,
) -> Path:
    """Train RGB-patch DiT on normal frame windows and return its run directory."""

    config = load_pipeline_config(config_path)
    active_run_id = run_id or config.training.run_id or default_run_id()
    active_max_steps = max_steps or config.training.max_steps
    if active_max_steps <= 0:
        raise ValueError("max_steps must be positive")
    active_overwrite = config.training.overwrite if overwrite is None else overwrite
    run_dir = config.training.output_root / active_run_id
    prepare_run_dir(run_dir, overwrite=active_overwrite)
    seed_everything(config.training.seed)

    records = load_sample_records(
        config.training.sample_index,
        normal_only=True,
        limit_samples=limit_samples,
    )
    train_records, val_records = split_train_val_records(
        records,
        val_fraction=config.training.val_fraction,
        seed=config.training.seed,
    )
    train_dataset = RGBPatchDataset(train_records, config.model)
    val_dataset = RGBPatchDataset(val_records, config.model) if val_records else None
    shape = infer_patch_shape(train_dataset[0], config.model)
    train_loader = make_loader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=True,
    )
    val_loader = (
        make_loader(
            val_dataset,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            shuffle=False,
        )
        if val_dataset is not None
        else None
    )

    device = get_device()
    configure_torch_backend()
    model = RGBPatchDiT(shape=shape, config=config.model).to(device)
    if config.training.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    flow = FlowMatcher(inference_steps=config.flow_matching.inference_steps)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled(config, device))
    metrics_path = run_dir / "metrics.jsonl"
    best_loss = float("inf")
    step = 0
    progress = progress_bar(total=active_max_steps, desc="train RGB patch DiT", unit="step")
    iterator = iter(train_loader)

    try:
        while step < active_max_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            model.train()
            context, target = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype(config.training.dtype),
                enabled=amp_enabled(config, device),
            ):
                noisy_target, time_values, velocity_target = flow.prepare_training_pair(target)
                prediction, memory_distance = model(noisy_target, time_values, context)
                velocity_loss = torch.nn.functional.mse_loss(prediction, velocity_target)
                loss = velocity_loss
            scaler.scale(loss).backward()
            if config.training.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            progress.update(1)

            if step % config.training.log_every_steps == 0 or step == 1:
                metric = {
                    "step": step,
                    "train_loss": float(loss.detach().cpu()),
                    "velocity_loss": float(velocity_loss.detach().cpu()),
                    "memory_distance": float(memory_distance.detach().mean().cpu()),
                }
                append_jsonl(metrics_path, metric)
                progress.set_postfix(loss=f"{metric['train_loss']:.6f}", refresh=False)

            should_save = step % config.training.save_every_steps == 0 or step == active_max_steps
            if should_save:
                val_loss = (
                    evaluate_loss(model, flow, val_loader, config, device) if val_loader else None
                )
                selected_loss = val_loss if val_loss is not None else float(loss.detach().cpu())
                is_best = selected_loss < best_loss
                best_loss = min(best_loss, selected_loss)
                save_checkpoint(
                    run_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    shape=shape,
                    model_config=config.model,
                    pipeline_config=config,
                    step=step,
                    best_loss=best_loss,
                )
                if is_best:
                    save_checkpoint(
                        run_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        shape=shape,
                        model_config=config.model,
                        pipeline_config=config,
                        step=step,
                        best_loss=best_loss,
                    )
                    progress_write(f"saved best checkpoint -> {run_dir / 'best.pt'}")
                append_jsonl(
                    metrics_path,
                    {
                        "step": step,
                        "validation_loss": val_loss,
                        "selected_loss": selected_loss,
                        "best_loss": best_loss,
                    },
                )
    finally:
        progress.close()
        cleanup_memory(cuda=device.type == "cuda")

    write_json(
        run_dir / "metrics_summary.json",
        {
            "run_id": active_run_id,
            "steps": step,
            "best_loss": best_loss,
            "checkpoint": str(run_dir / "best.pt"),
            "shape": asdict(shape),
        },
    )
    return run_dir


def evaluate_loss(
    model: nn.Module,
    flow: FlowMatcher,
    loader: Any,
    config: RGBPatchPipelineConfig,
    device: torch.device,
) -> float:
    """Estimate validation velocity MSE with fresh flow time/noise samples."""

    model.eval()
    values = []
    with torch.inference_mode():
        for batch in loader:
            context, target = move_batch(batch, device)
            noisy_target, time_values, velocity_target = flow.prepare_training_pair(target)
            prediction, _ = model(noisy_target, time_values, context)
            values.append(float(torch.nn.functional.mse_loss(prediction, velocity_target).cpu()))
    return float(np.mean(values)) if values else float("inf")


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    shape: RGBPatchShape,
    model_config: RGBPatchModelConfig,
    pipeline_config: RGBPatchPipelineConfig,
    step: int,
    best_loss: float,
) -> None:
    """Persist model/optimizer state with all shape-critical RGB patch metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "shape": asdict(shape),
            "model_config": asdict(model_config),
            "pipeline_config": config_to_dict(pipeline_config),
            "step": step,
            "best_loss": best_loss,
        },
        path,
    )


def load_checkpoint_for_inference(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[RGBPatchDiT, RGBPatchShape, RGBPatchModelConfig, dict[str, Any]]:
    """Restore an RGB-patch model and metadata for noise-to-patch inference."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {"model_state_dict", "shape", "model_config", "pipeline_config"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"RGB patch checkpoint is missing keys: {missing}")
    shape = RGBPatchShape(**checkpoint["shape"])
    model_config = RGBPatchModelConfig(**checkpoint["model_config"])
    model = RGBPatchDiT(shape=shape, config=model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, shape, model_config, checkpoint["pipeline_config"]


def move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Move direct RGB patch context/target tensors to the active device."""

    return (
        batch["context"].to(device, non_blocking=True),
        batch["target"].to(device, non_blocking=True),
    )


def amp_dtype(dtype: str) -> torch.dtype:
    """Map configured AMP dtype name to torch dtype."""

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


def amp_enabled(config: RGBPatchPipelineConfig, device: torch.device) -> bool:
    """Return whether CUDA automatic mixed precision is applicable."""

    return config.training.amp and device.type == "cuda" and config.training.dtype != "float32"


def configure_torch_backend() -> None:
    """Enable hardware-safe high-throughput PyTorch defaults."""

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when torch.compile wraps it."""

    return getattr(model, "_orig_mod", model)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch randomness."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_run_dir(run_dir: Path, *, overwrite: bool) -> None:
    """Create an empty run directory or reject accidental checkpoint mixing."""

    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Training run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a metrics record to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write indented UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_run_id() -> str:
    """Return a timestamped RGB patch training run identifier."""

    return datetime.now(timezone.utc).strftime("rgb_patch_dit_%Y%m%d_%H%M%S")
