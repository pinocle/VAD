"""Feature extraction cache pipeline for C_high, C_low, and Z."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.pipelines.feature_eng_pipeline import read_jsonl, write_jsonl
from src.utils import cleanup_memory, configure_progress_from_config, progress_bar

HIGH_TYPE_IMAGE = "image"
Z_TYPE_VAE = "vae"
LOW_TYPE_FRAME_DIFF = "frame_diff"
LOW_TYPE_OPTICAL_FLOW = "optical_flow"
TOKEN_MODES = {"cls", "patch", "all"}
Z_MODES = {"mode", "sample", "mean"}
FRAME_DIFF_MODES = {"signed", "abs"}
OPTICAL_FLOW_MODES = {"uv", "uv_mag"}
DTYPES = {"float16", "bfloat16", "float32"}
STAGES = {"high", "z", "low", "all"}
SPLITS = {"train", "test", "all"}


@dataclass(frozen=True)
class HighFeatureConfig:
    """Resolved C_high encoder feature settings."""

    type: str
    model_id: str
    image_size: int
    processor_do_resize: bool
    processor_do_center_crop: bool
    output_layer: int
    token_mode: str
    batch_size: int
    num_workers: int
    dtype: str
    freeze: bool
    cache: bool


@dataclass(frozen=True)
class ZFeatureConfig:
    """Resolved VAE Z feature settings."""

    type: str
    model_id: str
    image_size: int
    z_mode: str
    batch_size: int
    num_workers: int
    dtype: str
    freeze: bool
    cache: bool


@dataclass(frozen=True)
class LowFeatureConfig:
    """Resolved C_low feature settings."""

    type: str
    image_size: int
    mode: str
    method: str
    batch_size: int
    num_workers: int
    dtype: str
    cache: bool


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Resolved feature extraction configuration."""

    processed_root: Path
    feature_root: Path
    high: HighFeatureConfig
    z: ZFeatureConfig
    low: LowFeatureConfig

    def sample_path(self, split: str) -> Path:
        """Return the processed sample index path for a split."""

        if split == "train":
            return self.processed_root / "samples_train.jsonl"
        if split == "test":
            return self.processed_root / "samples_test.jsonl"
        raise ValueError("split must be 'train' or 'test'")

    def feature_index_path(self, split: str) -> Path:
        """Return the joined feature index path for a split."""

        if split == "train":
            return self.feature_root / "train_feature_index.jsonl"
        if split == "test":
            return self.feature_root / "test_feature_index.jsonl"
        raise ValueError("split must be 'train' or 'test'")


class HighFrameDataset(Dataset):
    """Dataset loading past frames for C_high extraction in DataLoader workers."""

    def __init__(self, records: list[dict[str, Any]], image_size: int) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        record["_high_images"] = [
            preprocess_image(path, self.image_size) for path in record["context_frame_paths"]
        ]
        return record


class ZFrameTensorDataset(Dataset):
    """Dataset loading VAE input tensors in DataLoader workers."""

    def __init__(self, records: list[dict[str, Any]], image_size: int) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        record["_z_frame_tensor"] = torch.stack(
            [load_vae_frame_tensor(path, self.image_size) for path in record["future_frame_paths"]]
        )
        return record


class LowFrameTensorDataset(Dataset):
    """Dataset loading past frames for simple C_low tensors."""

    def __init__(self, records: list[dict[str, Any]], image_size: int) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        record["_low_frame_tensor"] = torch.stack(
            [load_low_frame_tensor(path, self.image_size) for path in record["context_frame_paths"]]
        )
        return record


def load_config(path: Path) -> FeatureExtractionConfig:
    """Load feature extraction settings from the shared YAML config."""

    with path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}
    configure_progress_from_config(raw_config)
    dataset = raw_config.get("dataset", {})
    features = raw_config.get("features", {})
    if not features:
        raise ValueError("Missing features config section")

    batch_size = int(features.get("batch_size", 16))
    num_workers = int(features.get("num_workers", 4))
    dtype = str(features.get("dtype", "float16"))
    validate_common_options(batch_size, num_workers, dtype)

    high = load_high_config(features, batch_size, num_workers, dtype)
    z = load_z_config(features, batch_size, num_workers, dtype)
    low = load_low_config(features, batch_size, num_workers, dtype)

    return FeatureExtractionConfig(
        processed_root=Path(dataset["processed_root"]),
        feature_root=Path(features.get("root", "data/03_features/shanghaitech")),
        high=high,
        z=z,
        low=low,
    )


def validate_common_options(batch_size: int, num_workers: int, dtype: str) -> None:
    """Validate shared feature extraction settings."""

    if batch_size <= 0:
        raise ValueError("features.batch_size must be positive")
    if num_workers < 0:
        raise ValueError("features.num_workers must be non-negative")
    if dtype not in DTYPES:
        raise ValueError("features.dtype must be float16, bfloat16, or float32")


def load_high_config(
    features: dict[str, Any],
    batch_size: int,
    num_workers: int,
    dtype: str,
) -> HighFeatureConfig:
    """Resolve C_high encoder settings."""

    high = dict(features.get("high", {}))
    config = HighFeatureConfig(
        type=str(high.get("type", HIGH_TYPE_IMAGE)),
        model_id=str(high["model_id"]),
        image_size=int(high.get("image_size", 224)),
        processor_do_resize=bool(high.get("processor_do_resize", False)),
        processor_do_center_crop=bool(high.get("processor_do_center_crop", False)),
        output_layer=int(high.get("output_layer", -2)),
        token_mode=str(high.get("token_mode", "patch")),
        batch_size=int(high.get("batch_size", batch_size)),
        num_workers=int(high.get("num_workers", num_workers)),
        dtype=str(high.get("dtype", dtype)),
        freeze=bool(high.get("freeze", True)),
        cache=bool(high.get("cache", True)),
    )
    validate_high_config(config)
    return config


def validate_high_config(config: HighFeatureConfig) -> None:
    """Validate C_high encoder settings."""

    if config.type != HIGH_TYPE_IMAGE:
        raise ValueError("features.high.type='image' is the only supported v1 option")
    if config.token_mode not in TOKEN_MODES:
        raise ValueError("features.high.token_mode must be cls, patch, or all")
    validate_common_options(config.batch_size, config.num_workers, config.dtype)
    if config.cache and not config.freeze:
        raise ValueError("features.high.cache=true requires high.freeze=true")


def load_z_config(
    features: dict[str, Any],
    batch_size: int,
    num_workers: int,
    dtype: str,
) -> ZFeatureConfig:
    """Resolve VAE Z settings."""

    z = dict(features.get("z", {}))
    config = ZFeatureConfig(
        type=str(z.get("type", Z_TYPE_VAE)),
        model_id=str(z["model_id"]),
        image_size=int(z.get("image_size", 256)),
        z_mode=str(z.get("z_mode", "mode")),
        batch_size=int(z.get("batch_size", batch_size)),
        num_workers=int(z.get("num_workers", num_workers)),
        dtype=str(z.get("dtype", dtype)),
        freeze=bool(z.get("freeze", True)),
        cache=bool(z.get("cache", True)),
    )
    validate_z_config(config)
    return config


def validate_z_config(config: ZFeatureConfig) -> None:
    """Validate VAE Z settings."""

    if config.type != Z_TYPE_VAE:
        raise ValueError("features.z.type='vae' is the only supported v1 option")
    if config.z_mode not in Z_MODES:
        raise ValueError("features.z.z_mode must be mode, sample, or mean")
    validate_common_options(config.batch_size, config.num_workers, config.dtype)
    if config.cache and not config.freeze:
        raise ValueError("features.z.cache=true requires z.freeze=true")


def load_low_config(
    features: dict[str, Any],
    batch_size: int,
    num_workers: int,
    dtype: str,
) -> LowFeatureConfig:
    """Resolve C_low settings."""

    low = dict(features.get("low", {}))
    high = dict(features.get("high", {}))
    low_type = str(low.get("type", LOW_TYPE_FRAME_DIFF))
    config = LowFeatureConfig(
        type=low_type,
        image_size=int(low.get("image_size", high.get("image_size", 224))),
        mode=str(low.get("mode", "signed" if low_type == LOW_TYPE_FRAME_DIFF else "uv_mag")),
        method=str(low.get("method", "farneback")),
        batch_size=int(low.get("batch_size", batch_size)),
        num_workers=int(low.get("num_workers", num_workers)),
        dtype=str(low.get("dtype", dtype)),
        cache=bool(low.get("cache", True)),
    )
    validate_low_config(config)
    return config


def validate_low_config(config: LowFeatureConfig) -> None:
    """Validate C_low feature settings."""

    if config.type not in {LOW_TYPE_FRAME_DIFF, LOW_TYPE_OPTICAL_FLOW}:
        raise ValueError("features.low.type must be frame_diff or optical_flow")
    if config.type == LOW_TYPE_FRAME_DIFF and config.mode not in FRAME_DIFF_MODES:
        raise ValueError("features.low.mode must be signed or abs for frame_diff")
    if config.type == LOW_TYPE_OPTICAL_FLOW:
        if config.mode not in OPTICAL_FLOW_MODES:
            raise ValueError("features.low.mode must be uv or uv_mag for optical_flow")
        if config.method != "farneback":
            raise ValueError("features.low.method must be farneback for optical_flow")
    validate_common_options(config.batch_size, config.num_workers, config.dtype)


def torch_dtype(dtype: str) -> torch.dtype:
    """Return a torch dtype from a config dtype name."""

    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def model_load_kwargs(dtype: str, device: torch.device) -> dict[str, Any]:
    """Return optional HuggingFace model loading kwargs for the active device."""

    if device.type != "cuda":
        return {}
    return {"torch_dtype": torch_dtype(dtype)}


def autocast_is_enabled(dtype: torch.dtype, device: torch.device) -> bool:
    """Return whether CUDA autocast should be enabled for inference."""

    return device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}


def move_batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move a processor batch to the active device."""

    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def safe_signature(parts: dict[str, Any]) -> str:
    """Return a stable short signature for cache directory names."""

    normalized = "|".join(f"{key}={parts[key]}" for key in sorted(parts))
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    label = str(parts.get("model_id", "model")).replace("/", "_").replace(" ", "_")
    return f"{label}_{digest}"


def high_signature(config: HighFeatureConfig) -> str:
    """Return the C_high cache signature."""

    return safe_signature(
        {
            "model_id": config.model_id,
            "type": config.type,
            "image_size": config.image_size,
            "processor_do_resize": config.processor_do_resize,
            "processor_do_center_crop": config.processor_do_center_crop,
            "output_layer": config.output_layer,
            "token_mode": config.token_mode,
            "dtype": config.dtype,
        }
    )


def z_signature(config: ZFeatureConfig) -> str:
    """Return the VAE Z cache signature."""

    return safe_signature(
        {
            "model_id": config.model_id,
            "type": config.type,
            "image_size": config.image_size,
            "z_mode": config.z_mode,
            "dtype": config.dtype,
        }
    )


def low_signature(config: LowFeatureConfig) -> str:
    """Return the C_low cache signature."""

    return safe_signature(
        {
            "model_id": config.type,
            "type": config.type,
            "image_size": config.image_size,
            "mode": config.mode,
            "method": config.method,
            "dtype": config.dtype,
        }
    )


def high_cache_path(config: FeatureExtractionConfig, split: str, sample_id: str) -> Path:
    """Return one C_high feature cache path."""

    return config.feature_root / "high" / high_signature(config.high) / split / f"{sample_id}.pt"


def z_cache_path(config: FeatureExtractionConfig, split: str, sample_id: str) -> Path:
    """Return one VAE Z cache path."""

    return config.feature_root / "z" / z_signature(config.z) / split / f"{sample_id}.pt"


def low_cache_path(config: FeatureExtractionConfig, split: str, sample_id: str) -> Path:
    """Return one simple C_low feature cache path."""

    return config.feature_root / "low" / low_signature(config.low) / split / f"{sample_id}.pt"


def select_hidden_tokens(hidden_state: torch.Tensor, token_mode: str) -> torch.Tensor:
    """Select CLS, patch, or all tokens from a hidden-state tensor."""

    if hidden_state.ndim != 3:
        raise ValueError("hidden_state must be shaped [B, tokens, dim]")
    if token_mode == "all":
        return hidden_state
    if token_mode == "cls":
        return hidden_state[:, :1]
    if token_mode == "patch":
        if hidden_state.shape[1] <= 1:
            raise ValueError("patch token mode requires at least one non-CLS token")
        return hidden_state[:, 1:]
    raise ValueError("token_mode must be cls, patch, or all")


def select_high_layer(outputs: Any, output_layer: int) -> torch.Tensor:
    """Return one hidden-state layer from a HuggingFace model output."""

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise ValueError("Model output does not contain hidden_states")
    try:
        return hidden_states[output_layer]
    except IndexError as exc:
        raise ValueError(f"Invalid output_layer {output_layer}") from exc


def preprocess_image(path: str, image_size: int) -> Image.Image:
    """Load one frame as RGB PIL image."""

    image = Image.open(path).convert("RGB")
    return image.resize((image_size, image_size), Image.BICUBIC)


def load_vae_frame_tensor(path: str, image_size: int) -> torch.Tensor:
    """Load one frame normalized for diffusers AutoencoderKL."""

    image = preprocess_image(path, image_size)
    array = torch.from_numpy(np.asarray(image, dtype=np.float32))
    tensor = array.permute(2, 0, 1) / 127.5 - 1.0
    return tensor


def load_low_frame_tensor(path: str, image_size: int) -> torch.Tensor:
    """Load one RGB frame normalized to [0, 1] for frame differencing."""

    image = preprocess_image(path, image_size)
    array = torch.from_numpy(np.asarray(image, dtype=np.float32))
    return array.permute(2, 0, 1) / 255.0


def compute_low_tensor(
    frames: torch.Tensor,
    mode: str,
    *,
    low_type: str = LOW_TYPE_FRAME_DIFF,
    method: str = "farneback",
) -> torch.Tensor:
    """Return C_low shaped [T-1, C, H, W]."""

    if frames.ndim != 4:
        raise ValueError("low frames must be shaped [T, C, H, W]")
    if frames.shape[0] < 2:
        raise ValueError("C_low features require at least two past frames")
    if low_type == LOW_TYPE_OPTICAL_FLOW:
        return compute_optical_flow_tensor(frames, mode=mode, method=method)
    if low_type != LOW_TYPE_FRAME_DIFF:
        raise ValueError("features.low.type must be frame_diff or optical_flow")
    if mode not in FRAME_DIFF_MODES:
        raise ValueError("features.low.mode must be signed or abs for frame_diff")
    low = frames[:-1] - frames[1:]
    return low.abs() if mode == "abs" else low


def compute_optical_flow_tensor(
    frames: torch.Tensor,
    *,
    mode: str,
    method: str,
) -> torch.Tensor:
    """Compute Farneback optical flow C_low from normalized RGB frames."""

    if method != "farneback":
        raise ValueError("features.low.method must be farneback for optical_flow")
    if mode not in OPTICAL_FLOW_MODES:
        raise ValueError("features.low.mode must be uv or uv_mag for optical_flow")

    import cv2

    frames_np = frames.detach().cpu().permute(0, 2, 3, 1).numpy()
    grays = []
    for frame in frames_np:
        gray = np.clip(
            (0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]) * 255.0,
            0,
            255,
        ).astype(np.uint8)
        grays.append(gray)

    height, width = grays[0].shape
    scale = float(max(height, width))
    flows = []
    for previous, current in zip(grays[:-1], grays[1:], strict=True):
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flow = flow.astype(np.float32) / scale
        channels = [flow[..., 0], flow[..., 1]]
        if mode == "uv_mag":
            channels.append(np.linalg.norm(flow, axis=2))
        flows.append(torch.from_numpy(np.stack(channels, axis=0)))
    return torch.stack(flows)


def sample_collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return sample records unchanged."""

    return batch


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Build a DataLoader over sample records."""

    options: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": sample_collate,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 4
    return DataLoader(
        dataset,
        **options,
    )


def pending_high_records(
    config: FeatureExtractionConfig,
    split: str,
    records: list[dict[str, Any]],
    overwrite: bool,
) -> list[dict[str, Any]]:
    """Return records whose C_high feature cache should be written."""

    return [
        sample
        for sample in records
        if overwrite or not high_cache_path(config, split, sample["sample_id"]).is_file()
    ]


def pending_z_records(
    config: FeatureExtractionConfig,
    split: str,
    records: list[dict[str, Any]],
    overwrite: bool,
) -> list[dict[str, Any]]:
    """Return records whose VAE Z cache should be written."""

    return [
        sample
        for sample in records
        if overwrite or not z_cache_path(config, split, sample["sample_id"]).is_file()
    ]


def pending_low_records(
    config: FeatureExtractionConfig,
    split: str,
    records: list[dict[str, Any]],
    overwrite: bool,
) -> list[dict[str, Any]]:
    """Return records whose simple C_low cache should be written."""

    return [
        sample
        for sample in records
        if overwrite or not low_cache_path(config, split, sample["sample_id"]).is_file()
    ]


def cache_high_features(
    config: FeatureExtractionConfig,
    *,
    split: str,
    records: list[dict[str, Any]],
    overwrite: bool,
    limit_samples: int | None,
) -> None:
    """Cache frozen HuggingFace image encoder features as C_high."""

    if not config.high.cache:
        return
    if not config.high.freeze:
        raise ValueError("C_high feature caching requires high.freeze=true")

    from transformers import AutoImageProcessor, AutoModel

    records = pending_high_records(
        config,
        split,
        limited_records(records, limit_samples),
        overwrite,
    )
    if not records:
        return

    device = get_device()
    dtype = torch_dtype(config.high.dtype)
    processor = AutoImageProcessor.from_pretrained(config.high.model_id)
    model = AutoModel.from_pretrained(
        config.high.model_id,
        **model_load_kwargs(config.high.dtype, device),
    ).to(device)
    model.eval()

    try:
        loader = make_loader(
            HighFrameDataset(records, config.high.image_size),
            batch_size=config.high.batch_size,
            num_workers=config.high.num_workers,
        )
        for batch in progress_bar(
            loader,
            desc=f"{split} C_high features",
            unit="batch",
            leave=False,
        ):
            images = [image for sample in batch for image in sample["_high_images"]]
            inputs = processor(
                images=images,
                return_tensors="pt",
                do_resize=config.high.processor_do_resize,
                do_center_crop=config.high.processor_do_center_crop,
            )
            inputs = move_batch_to_device(inputs, device)
            with torch.inference_mode():
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=autocast_is_enabled(dtype, device),
                ):
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                    hidden = select_high_layer(outputs, config.high.output_layer)
                    tokens = select_hidden_tokens(hidden, config.high.token_mode)

            cursor = 0
            for sample in batch:
                frame_count = len(sample["context_frame_paths"])
                embedding = tokens[cursor : cursor + frame_count].detach().cpu().to(dtype)
                cursor += frame_count
                payload = {
                    "sample_id": sample["sample_id"],
                    "high": embedding,
                    "model_id": config.high.model_id,
                    "type": config.high.type,
                    "image_size": config.high.image_size,
                    "processor_do_resize": config.high.processor_do_resize,
                    "processor_do_center_crop": config.high.processor_do_center_crop,
                    "output_layer": config.high.output_layer,
                    "token_mode": config.high.token_mode,
                    "frame_paths": sample["context_frame_paths"],
                }
                save_torch_payload(high_cache_path(config, split, sample["sample_id"]), payload)
    finally:
        del model, processor
        cleanup_memory(cuda=device.type == "cuda")


def cache_z_features(
    config: FeatureExtractionConfig,
    *,
    split: str,
    records: list[dict[str, Any]],
    overwrite: bool,
    limit_samples: int | None,
) -> None:
    """Cache frozen VAE Z tensors for future frames."""

    if not config.z.cache:
        return
    if not config.z.freeze:
        raise ValueError("Z caching requires z.freeze=true")

    from diffusers.models import AutoencoderKL

    records = pending_z_records(
        config,
        split,
        limited_records(records, limit_samples),
        overwrite,
    )
    if not records:
        return

    device = get_device()
    dtype = torch_dtype(config.z.dtype)
    vae = AutoencoderKL.from_pretrained(
        config.z.model_id,
        **model_load_kwargs(config.z.dtype, device),
    ).to(device)
    vae.eval()
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))

    try:
        loader = make_loader(
            ZFrameTensorDataset(records, config.z.image_size),
            batch_size=config.z.batch_size,
            num_workers=config.z.num_workers,
        )
        for batch in progress_bar(loader, desc=f"{split} Z features", unit="batch", leave=False):
            frame_tensor = torch.cat([sample["_z_frame_tensor"] for sample in batch], dim=0)
            frame_tensor = frame_tensor.to(device, non_blocking=True)
            with torch.inference_mode():
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=autocast_is_enabled(dtype, device),
                ):
                    distribution = vae.encode(frame_tensor).latent_dist
                    z_values = select_vae_z(distribution, config.z.z_mode) * scaling_factor

            cursor = 0
            for sample in batch:
                frame_count = len(sample["future_frame_paths"])
                z = z_values[cursor : cursor + frame_count].detach().cpu().to(dtype)
                cursor += frame_count
                payload = {
                    "sample_id": sample["sample_id"],
                    "z": z,
                    "model_id": config.z.model_id,
                    "type": config.z.type,
                    "z_mode": config.z.z_mode,
                    "scaling_factor": scaling_factor,
                    "frame_paths": sample["future_frame_paths"],
                }
                save_torch_payload(z_cache_path(config, split, sample["sample_id"]), payload)
    finally:
        del vae
        cleanup_memory(cuda=device.type == "cuda")


def cache_low_features(
    config: FeatureExtractionConfig,
    *,
    split: str,
    records: list[dict[str, Any]],
    overwrite: bool,
    limit_samples: int | None,
) -> None:
    """Cache simple frame-difference C_low features from past frames."""

    if not config.low.cache:
        return

    records = pending_low_records(
        config,
        split,
        limited_records(records, limit_samples),
        overwrite,
    )
    if not records:
        return

    dtype = torch_dtype(config.low.dtype)
    try:
        loader = make_loader(
            LowFrameTensorDataset(records, config.low.image_size),
            batch_size=config.low.batch_size,
            num_workers=config.low.num_workers,
        )
        for batch in progress_bar(
            loader,
            desc=f"{split} C_low features",
            unit="batch",
            leave=False,
        ):
            for sample in batch:
                low = compute_low_tensor(
                    sample["_low_frame_tensor"],
                    config.low.mode,
                    low_type=config.low.type,
                    method=config.low.method,
                ).to(dtype)
                payload = {
                    "sample_id": sample["sample_id"],
                    "low": low,
                    "type": config.low.type,
                    "mode": config.low.mode,
                    "method": config.low.method,
                    "frame_paths": sample["context_frame_paths"],
                }
                save_torch_payload(low_cache_path(config, split, sample["sample_id"]), payload)
    finally:
        cleanup_memory(cuda=False)


def select_vae_z(distribution: Any, z_mode: str) -> torch.Tensor:
    """Select VAE Z mode/sample/mean from a diffusers distribution."""

    if z_mode == "mode":
        return distribution.mode()
    if z_mode == "sample":
        return distribution.sample()
    if z_mode == "mean":
        return distribution.mean
    raise ValueError("z_mode must be mode, sample, or mean")


def save_torch_payload(path: Path, payload: dict[str, Any]) -> None:
    """Save a torch payload, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def limited_records(
    records: list[dict[str, Any]],
    limit_samples: int | None,
) -> list[dict[str, Any]]:
    """Return optionally limited records."""

    if limit_samples is None:
        return records
    if limit_samples <= 0:
        raise ValueError("limit_samples must be positive")
    return records[:limit_samples]


def build_feature_index_records(
    config: FeatureExtractionConfig,
    split: str,
    records: list[dict[str, Any]],
    *,
    limit_samples: int | None,
) -> list[dict[str, Any]]:
    """Build joined records linking samples to cached C_high/Z/C_low tensors."""

    index_records = []
    for sample in limited_records(records, limit_samples):
        record = dict(sample)
        if config.high.cache:
            record["high_feature_path"] = str(high_cache_path(config, split, sample["sample_id"]))
        if config.z.cache:
            record["z_path"] = str(z_cache_path(config, split, sample["sample_id"]))
        if config.low.cache:
            record["low_feature_path"] = str(low_cache_path(config, split, sample["sample_id"]))
        index_records.append(record)
    return index_records


def write_feature_index(
    config: FeatureExtractionConfig,
    split: str,
    records: list[dict[str, Any]],
    *,
    limit_samples: int | None,
) -> None:
    """Write the joined feature index for one split."""

    write_jsonl(
        config.feature_index_path(split),
        build_feature_index_records(config, split, records, limit_samples=limit_samples),
    )


def run_extract_features(
    config: FeatureExtractionConfig,
    *,
    split: str,
    stage: str,
    limit_samples: int | None = None,
    overwrite: bool = False,
) -> None:
    """Run C_high/Z/C_low feature extraction for one or more splits."""

    if split not in SPLITS:
        raise ValueError("split must be train, test, or all")
    if stage not in STAGES:
        raise ValueError("stage must be high, z, low, or all")

    active_splits = ["train", "test"] if split == "all" else [split]
    active_stages = [name for name in ("high", "z", "low") if stage in {name, "all"}]
    stage_progress = progress_bar(
        total=len(active_splits) * len(active_stages),
        desc="feature extraction stages",
        unit="stage",
    )
    try:
        for active_split in active_splits:
            sample_path = config.sample_path(active_split)
            if not sample_path.is_file():
                raise FileNotFoundError(
                    f"Missing processed sample index: {sample_path}. Run preprocessing first."
                )
            records = read_jsonl(sample_path)
            for active_stage in active_stages:
                stage_progress.set_postfix(
                    split=active_split,
                    stage=active_stage,
                    refresh=False,
                )
                if active_stage == "high":
                    cache_high_features(
                        config,
                        split=active_split,
                        records=records,
                        overwrite=overwrite,
                        limit_samples=limit_samples,
                    )
                elif active_stage == "z":
                    cache_z_features(
                        config,
                        split=active_split,
                        records=records,
                        overwrite=overwrite,
                        limit_samples=limit_samples,
                    )
                elif active_stage == "low":
                    cache_low_features(
                        config,
                        split=active_split,
                        records=records,
                        overwrite=overwrite,
                        limit_samples=limit_samples,
                    )
                stage_progress.update(1)
            write_feature_index(
                config,
                active_split,
                records,
                limit_samples=limit_samples,
            )
            print(f"wrote feature index -> {config.feature_index_path(active_split)}")
    finally:
        stage_progress.close()


def get_device() -> torch.device:
    """Return the active torch device."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse feature extraction CLI arguments."""

    parser = argparse.ArgumentParser(description="Extract cached C_high/C_low/Z features.")
    parser.add_argument("--config", type=Path, default=Path("config/local.yaml"))
    parser.add_argument("--split", choices=sorted(SPLITS), default="all")
    parser.add_argument("--stage", choices=sorted(STAGES), default="all")
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""

    args = parse_args(argv)
    config = load_config(args.config)
    run_extract_features(
        config,
        split=args.split,
        stage=args.stage,
        limit_samples=args.limit_samples,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
