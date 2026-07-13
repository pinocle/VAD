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
from PIL import Image, ImageDraw
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
    unpatchify_rgb,
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
    """Train RGB-patch DiT on all normal train windows and return its run directory."""

    config = load_pipeline_config(config_path)
    active_run_id = run_id or config.training.run_id or default_run_id()
    active_overwrite = config.training.overwrite if overwrite is None else overwrite
    run_dir = config.training.output_root / active_run_id
    prepare_run_dir(run_dir, overwrite=active_overwrite)
    seed_everything(config.training.seed)

    records = load_sample_records(
        config.training.sample_index,
        normal_only=True,
        limit_samples=limit_samples,
    )
    train_dataset = RGBPatchDataset(records, config.model)
    shape = infer_patch_shape(train_dataset[0], config.model)
    train_loader = make_loader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=True,
    )
    steps_per_epoch = len(train_loader)
    configured_steps = steps_per_epoch * config.training.epochs
    active_max_steps = max_steps if max_steps is not None else configured_steps
    if active_max_steps <= 0:
        raise ValueError("max_steps must be positive")

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
    epoch_losses: list[float] = []
    preview = build_fixed_preview(train_dataset[0], seed=config.training.seed)
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
            loss_value = float(loss.detach().cpu())
            epoch_losses.append(loss_value)
            progress.update(1)

            if step % config.training.log_every_steps == 0 or step == 1:
                metric = {
                    "step": step,
                    "epoch": (step - 1) // steps_per_epoch + 1,
                    "train_loss": loss_value,
                    "velocity_loss": float(velocity_loss.detach().cpu()),
                    "memory_distance": float(memory_distance.detach().mean().cpu()),
                }
                append_jsonl(metrics_path, metric)
                progress.set_postfix(loss=f"{metric['train_loss']:.6f}", refresh=False)

            should_preview = config.training.preview_every_steps > 0 and (
                step % config.training.preview_every_steps == 0 or step == active_max_steps
            )
            if should_preview:
                preview_metrics = save_generation_preview(
                    model,
                    flow,
                    preview,
                    shape=shape,
                    device=device,
                    output_path=run_dir / "previews" / f"step_{step:08d}.png",
                    step=step,
                    inference_steps=config.flow_matching.inference_steps,
                )
                append_jsonl(metrics_path, preview_metrics)

            epoch_boundary = step % steps_per_epoch == 0 or step == active_max_steps
            if epoch_boundary:
                epoch_train_loss = float(np.mean(epoch_losses))
                epoch_index = (step - 1) // steps_per_epoch + 1
                is_best = epoch_train_loss < best_loss
                best_loss = min(best_loss, epoch_train_loss)
                append_jsonl(
                    metrics_path,
                    {
                        "step": step,
                        "epoch": epoch_index,
                        "epoch_train_loss": epoch_train_loss,
                        "best_loss": best_loss,
                    },
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
                epoch_losses.clear()

            should_save = step % config.training.save_every_steps == 0 or step == active_max_steps
            if should_save:
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
    finally:
        progress.close()
        cleanup_memory(cuda=device.type == "cuda")

    write_json(
        run_dir / "metrics_summary.json",
        {
            "run_id": active_run_id,
            "steps": step,
            "steps_per_epoch": steps_per_epoch,
            "configured_epochs": config.training.epochs,
            "completed_epochs": step / steps_per_epoch,
            "training_samples": len(train_dataset),
            "best_loss": best_loss,
            "checkpoint": str(run_dir / "best.pt"),
            "shape": asdict(shape),
        },
    )
    return run_dir


def build_fixed_preview(sample: dict[str, Any], *, seed: int) -> dict[str, torch.Tensor]:
    """Build one fixed train sample/noise pair for comparable generation snapshots."""

    context = sample["context"].unsqueeze(0).cpu()
    target = sample["target"].unsqueeze(0).cpu()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_noise = torch.randn(target.shape, generator=generator, dtype=target.dtype)
    return {
        "context": context,
        "target": target,
        "initial_noise": initial_noise,
    }


@torch.inference_mode()
def save_generation_preview(
    model: nn.Module,
    flow: FlowMatcher,
    preview: dict[str, torch.Tensor],
    *,
    shape: RGBPatchShape,
    device: torch.device,
    output_path: Path,
    step: int,
    inference_steps: int,
) -> dict[str, Any]:
    """Save a fixed-seed context/target/generated/error preview during training."""

    was_training = model.training
    model.eval()
    try:
        context = preview["context"].to(device)
        target = preview["target"].to(device)
        generated, memory_distance = flow.sample(
            model,
            context=context,
            target_shape=tuple(target.shape),
            inference_steps=inference_steps,
            initial_noise=preview["initial_noise"],
        )
    finally:
        if was_training:
            model.train()

    context_rgb = unpatchify_rgb(
        context[0].cpu(),
        image_size=shape.image_size,
        patch_size=shape.patch_size,
    )
    target_rgb = unpatchify_rgb(
        target[0].cpu(),
        image_size=shape.image_size,
        patch_size=shape.patch_size,
    )
    generated_rgb = unpatchify_rgb(
        generated[0].cpu(),
        image_size=shape.image_size,
        patch_size=shape.patch_size,
    )
    context_image = rgb_frame_to_array(context_rgb[-1])
    target_image = rgb_frame_to_array(target_rgb[0])
    generated_image = rgb_frame_to_array(generated_rgb[0])
    error_image = rgb_error_to_array(target_image, generated_image)

    labels = ("last context", "actual future +1", "generated future +1", "RGB error")
    images = (context_image, target_image, generated_image, error_image)
    banner_height = 24
    canvas = Image.new(
        "RGB",
        (shape.image_size * len(images), shape.image_size + banner_height),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(zip(labels, images, strict=True)):
        x_offset = index * shape.image_size
        draw.text((x_offset + 4, 5), label, fill="black")
        canvas.paste(Image.fromarray(image), (x_offset, banner_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    metrics = {
        "step": step,
        "preview_path": str(output_path),
        "preview_mse": float(torch.nn.functional.mse_loss(generated.float(), target.float()).cpu()),
        "preview_generated_std": float(generated.float().std().cpu()),
        "preview_generated_min": float(generated.float().min().cpu()),
        "preview_generated_max": float(generated.float().max().cpu()),
        "preview_memory_distance": float(memory_distance.float().mean().cpu()),
    }
    write_json(output_path.with_suffix(".json"), metrics)
    return metrics


def rgb_frame_to_array(frame: torch.Tensor) -> np.ndarray:
    """Convert one normalized CHW RGB frame into a displayable uint8 array."""

    return (
        frame.float().add(1.0).div(2.0).clamp(0.0, 1.0).mul(255.0).byte().permute(1, 2, 0).numpy()
    )


def rgb_error_to_array(target: np.ndarray, generated: np.ndarray) -> np.ndarray:
    """Render mean absolute RGB error as a dark-to-yellow heatmap."""

    error = np.abs(target.astype(np.float32) - generated.astype(np.float32)).mean(axis=-1) / 255.0
    red = np.clip(error * 2.0, 0.0, 1.0)
    green = np.clip((error - 0.25) * 2.0, 0.0, 1.0)
    blue = np.zeros_like(error)
    return (np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)


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
