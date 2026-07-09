"""Detector/tracker cache conversion for track-aware object grids."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import torch

from src.pipelines.feature_eng_pipeline import read_jsonl, write_jsonl

TRACK_CHANNELS = ("objectness", "velocity_x", "velocity_y", "speed", "trajectory")
TRAJECTORY_MODE_CAUSAL = "causal"
TRAJECTORY_MODE_WINDOW = "window"
TRAJECTORY_MODES = {TRAJECTORY_MODE_CAUSAL, TRAJECTORY_MODE_WINDOW}


@dataclass(frozen=True)
class TrackObject:
    """One validated detector/tracker object in normalized image coordinates."""

    video_id: str
    frame_idx: int
    track_id: str
    bbox_xyxy_norm: tuple[float, float, float, float]
    score: float
    class_id: int | None = None
    class_name: str | None = None
    velocity_xy_norm: tuple[float, float] | None = None
    age: int | None = None
    is_interpolated: bool = False

    @property
    def center_xy(self) -> tuple[float, float]:
        """Return the normalized bbox center."""

        x1, y1, x2, y2 = self.bbox_xyxy_norm
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def velocity(self) -> tuple[float, float]:
        """Return velocity, after fallback filling has run."""

        return self.velocity_xy_norm or (0.0, 0.0)


@dataclass(frozen=True)
class TrackFrame:
    """One raw JSONL frame record after schema validation."""

    video_id: str
    frame_idx: int
    objects: tuple[TrackObject, ...]
    image_size: tuple[int, int] | None = None


def load_raw_track_cache(path: Path) -> dict[tuple[str, int], tuple[TrackObject, ...]]:
    """Load, validate, and velocity-fill a raw detector/tracker JSONL cache."""

    frames = []
    for jsonl_path in collect_jsonl_paths(path):
        for line_number, raw in enumerate(read_jsonl(jsonl_path), start=1):
            try:
                frames.append(parse_track_frame(raw))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid track cache record {jsonl_path}:{line_number}: {error}"
                ) from error

    objects = fill_missing_velocities([obj for frame in frames for obj in frame.objects])
    objects_by_frame: dict[tuple[str, int], list[TrackObject]] = {}
    for obj in objects:
        objects_by_frame.setdefault((obj.video_id, obj.frame_idx), []).append(obj)
    return {
        key: tuple(sorted(value, key=lambda item: item.track_id))
        for key, value in objects_by_frame.items()
    }


def collect_jsonl_paths(path: Path) -> list[Path]:
    """Return sorted JSONL files from one file or directory path."""

    if path.is_file():
        return [path]
    if path.is_dir():
        paths = sorted(item for item in path.rglob("*.jsonl") if item.is_file())
        if paths:
            return paths
    raise FileNotFoundError(f"Track cache path must be a JSONL file or directory: {path}")


def parse_track_frame(raw: dict[str, Any]) -> TrackFrame:
    """Validate one raw detector/tracker frame record."""

    if not isinstance(raw, dict):
        raise ValueError("frame record must be a JSON object")
    video_id = require_str(raw, "video_id")
    frame_idx = require_int(raw, "frame_idx")
    objects_value = raw.get("objects")
    if not isinstance(objects_value, list):
        raise ValueError("objects must be a list")
    image_size = parse_image_size(raw.get("image_size"))
    objects = tuple(parse_track_object(video_id, frame_idx, item) for item in objects_value)
    return TrackFrame(
        video_id=video_id,
        frame_idx=frame_idx,
        objects=objects,
        image_size=image_size,
    )


def parse_track_object(video_id: str, frame_idx: int, raw: dict[str, Any]) -> TrackObject:
    """Validate one raw object record."""

    if not isinstance(raw, dict):
        raise ValueError("each object must be a JSON object")
    track_id = require_str(raw, "track_id")
    bbox = parse_bbox_xyxy_norm(raw.get("bbox_xyxy_norm"))
    score = require_float(raw, "score")
    if not 0.0 <= score <= 1.0:
        raise ValueError("object score must be in [0, 1]")
    return TrackObject(
        video_id=video_id,
        frame_idx=frame_idx,
        track_id=track_id,
        bbox_xyxy_norm=bbox,
        score=score,
        class_id=parse_optional_int(raw.get("class_id")),
        class_name=parse_optional_str(raw.get("class_name")),
        velocity_xy_norm=parse_optional_velocity(raw.get("velocity_xy_norm")),
        age=parse_optional_int(raw.get("age")),
        is_interpolated=bool(raw.get("is_interpolated", False)),
    )


def require_str(raw: dict[str, Any], key: str) -> str:
    """Return a required non-empty string field."""

    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def require_int(raw: dict[str, Any], key: str) -> int:
    """Return a required integer field."""

    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def require_float(raw: dict[str, Any], key: str) -> float:
    """Return a required numeric field."""

    value = raw.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def parse_optional_str(value: Any) -> str | None:
    """Parse an optional string."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional string fields must be strings")
    return value


def parse_optional_int(value: Any) -> int | None:
    """Parse an optional integer."""

    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("optional integer fields must be integers")
    return value


def parse_image_size(value: Any) -> tuple[int, int] | None:
    """Parse optional ``[height, width]`` image size metadata."""

    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("image_size must be [height, width] when present")
    height, width = value
    if (
        not isinstance(height, int)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or isinstance(width, bool)
        or height <= 0
        or width <= 0
    ):
        raise ValueError("image_size values must be positive integers")
    return (height, width)


def parse_bbox_xyxy_norm(value: Any) -> tuple[float, float, float, float]:
    """Parse and validate normalized ``xyxy`` bbox coordinates."""

    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bbox_xyxy_norm must be a list of four numbers")
    coords = []
    for coord in value:
        if not isinstance(coord, (int, float)) or isinstance(coord, bool):
            raise ValueError("bbox_xyxy_norm values must be numeric")
        coords.append(float(coord))
    x1, y1, x2, y2 = coords
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("bbox_xyxy_norm must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return (x1, y1, x2, y2)


def parse_optional_velocity(value: Any) -> tuple[float, float] | None:
    """Parse optional normalized velocity metadata."""

    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("velocity_xy_norm must be [vx, vy] when present")
    vx, vy = value
    if (
        not isinstance(vx, (int, float))
        or isinstance(vx, bool)
        or not isinstance(vy, (int, float))
        or isinstance(vy, bool)
    ):
        raise ValueError("velocity_xy_norm values must be numeric")
    return (float(vx), float(vy))


def fill_missing_velocities(objects: list[TrackObject]) -> list[TrackObject]:
    """Fill missing velocities from adjacent same-video same-track centers."""

    grouped: dict[tuple[str, str], list[TrackObject]] = {}
    for obj in objects:
        grouped.setdefault((obj.video_id, obj.track_id), []).append(obj)

    filled: list[TrackObject] = []
    for track_objects in grouped.values():
        ordered = sorted(track_objects, key=lambda item: item.frame_idx)
        for index, obj in enumerate(ordered):
            if obj.velocity_xy_norm is not None:
                filled.append(obj)
                continue
            previous = ordered[index - 1] if index > 0 else None
            following = ordered[index + 1] if index + 1 < len(ordered) else None
            if previous is not None:
                velocity = center_delta(previous, obj)
            elif following is not None:
                velocity = center_delta(obj, following)
            else:
                velocity = (0.0, 0.0)
            filled.append(replace(obj, velocity_xy_norm=velocity))
    return sorted(filled, key=lambda item: (item.video_id, item.frame_idx, item.track_id))


def center_delta(first: TrackObject, second: TrackObject) -> tuple[float, float]:
    """Return center displacement per frame from ``first`` to ``second``."""

    frame_delta = max(1, int(second.frame_idx) - int(first.frame_idx))
    x1, y1 = first.center_xy
    x2, y2 = second.center_xy
    return ((x2 - x1) / frame_delta, (y2 - y1) / frame_delta)


def rasterize_sample_grids(
    sample: dict[str, Any],
    objects_by_frame: dict[tuple[str, int], tuple[TrackObject, ...]],
    *,
    grid_size: tuple[int, int],
    trajectory_mode: str = TRAJECTORY_MODE_CAUSAL,
) -> dict[str, torch.Tensor | dict[str, Any]]:
    """Build dense context/future grids for one feature-index sample."""

    if trajectory_mode not in TRAJECTORY_MODES:
        raise ValueError("trajectory_mode must be causal or window")
    video_id = str(sample["video_id"])
    context_frames = [int(frame_idx) for frame_idx in sample["context_frames"]]
    future_frames = [int(frame_idx) for frame_idx in sample["future_frames"]]
    window_frames = context_frames + future_frames
    frame_objects = {
        frame_idx: tuple(objects_by_frame.get((video_id, frame_idx), ()))
        for frame_idx in window_frames
    }

    context_grid = torch.stack(
        [
            rasterize_frame_grid(
                frame_objects[frame_idx],
                grid_size=grid_size,
                trajectory_objects=trajectory_objects_for_frame(
                    frame_idx,
                    window_frames=window_frames,
                    frame_objects=frame_objects,
                    trajectory_mode=trajectory_mode,
                ),
            )
            for frame_idx in context_frames
        ]
    )
    future_grid = torch.stack(
        [
            rasterize_frame_grid(
                frame_objects[frame_idx],
                grid_size=grid_size,
                trajectory_objects=trajectory_objects_for_frame(
                    frame_idx,
                    window_frames=window_frames,
                    frame_objects=frame_objects,
                    trajectory_mode=trajectory_mode,
                ),
            )
            for frame_idx in future_frames
        ]
    )
    return {
        "context_grid": context_grid,
        "future_grid": future_grid,
        "metadata": {
            "sample_id": sample["sample_id"],
            "video_id": video_id,
            "context_frames": context_frames,
            "future_frames": future_frames,
            "grid_size": list(grid_size),
            "channels": list(TRACK_CHANNELS),
            "trajectory_mode": trajectory_mode,
            "trajectory_mode_note": (
                "causal uses sampled-window track history up to each scoring frame; "
                "window uses the full sampled window and is offline-analysis only"
            ),
        },
    }


def trajectory_objects_for_frame(
    frame_idx: int,
    *,
    window_frames: list[int],
    frame_objects: dict[int, tuple[TrackObject, ...]],
    trajectory_mode: str,
) -> list[TrackObject]:
    """Return objects contributing to the trajectory channel for one frame."""

    if trajectory_mode == TRAJECTORY_MODE_WINDOW:
        active_frames = window_frames
    else:
        active_frames = [candidate for candidate in window_frames if candidate <= frame_idx]
    return [obj for active_frame in active_frames for obj in frame_objects[active_frame]]


def rasterize_frame_grid(
    objects: Iterable[TrackObject],
    *,
    grid_size: tuple[int, int],
    trajectory_objects: Iterable[TrackObject] = (),
) -> torch.Tensor:
    """Rasterize one frame to ``[channels, H, W]`` dense track grid."""

    height, width = validate_grid_size(grid_size)
    grid = torch.zeros(len(TRACK_CHANNELS), height, width, dtype=torch.float32)
    for obj in objects:
        y_start, y_stop, x_start, x_stop = bbox_grid_bounds(obj.bbox_xyxy_norm, grid_size)
        current = grid[0, y_start:y_stop, x_start:x_stop]
        mask = obj.score >= current
        if not bool(mask.any()):
            continue
        vx, vy = obj.velocity
        speed = math.sqrt(vx * vx + vy * vy)
        grid[0, y_start:y_stop, x_start:x_stop] = torch.where(
            mask,
            torch.full_like(current, obj.score),
            current,
        )
        for channel, value in ((1, vx), (2, vy), (3, speed)):
            region = grid[channel, y_start:y_stop, x_start:x_stop]
            grid[channel, y_start:y_stop, x_start:x_stop] = torch.where(
                mask,
                torch.full_like(region, float(value)),
                region,
            )

    for obj in trajectory_objects:
        x, y = obj.center_xy
        row = min(height - 1, max(0, int(math.floor(y * height))))
        col = min(width - 1, max(0, int(math.floor(x * width))))
        grid[4, row, col] = max(float(grid[4, row, col]), obj.score)
    return grid


def validate_grid_size(grid_size: tuple[int, int]) -> tuple[int, int]:
    """Validate and return ``(height, width)`` grid size."""

    height, width = grid_size
    if height <= 0 or width <= 0:
        raise ValueError("grid_size must contain positive height and width")
    return int(height), int(width)


def bbox_grid_bounds(
    bbox_xyxy_norm: tuple[float, float, float, float],
    grid_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return inclusive-exclusive row/column bounds covered by a bbox."""

    height, width = validate_grid_size(grid_size)
    x1, y1, x2, y2 = bbox_xyxy_norm
    x_start = min(width - 1, max(0, int(math.floor(x1 * width))))
    y_start = min(height - 1, max(0, int(math.floor(y1 * height))))
    x_stop = min(width, max(x_start + 1, int(math.ceil(x2 * width))))
    y_stop = min(height, max(y_start + 1, int(math.ceil(y2 * height))))
    return y_start, y_stop, x_start, x_stop


def save_track_feature_payload(path: Path, payload: dict[str, Any]) -> None:
    """Save one per-sample track feature payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_track_feature_payload(path: Path) -> dict[str, Any]:
    """Load and validate one per-sample track feature payload."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing track feature tensor: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Track feature payload must be a dictionary: {path}")
    for key in ("context_grid", "future_grid", "metadata"):
        if key not in payload:
            raise ValueError(f"Track feature payload is missing {key}: {path}")
    for key in ("context_grid", "future_grid"):
        value = payload[key]
        if not isinstance(value, torch.Tensor) or value.ndim != 4:
            raise ValueError(f"{key} must be a tensor shaped [T, C, H, W]: {path}")
        if value.shape[1] != len(TRACK_CHANNELS):
            raise ValueError(
                f"{key} must have {len(TRACK_CHANNELS)} channels {TRACK_CHANNELS}: {path}"
            )
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata must be a dictionary: {path}")
    grid_size = metadata.get("grid_size")
    if not isinstance(grid_size, list) or len(grid_size) != 2:
        raise ValueError(f"metadata.grid_size must be [height, width]: {path}")
    for key in ("context_grid", "future_grid"):
        value = payload[key]
        if list(value.shape[-2:]) != grid_size:
            raise ValueError(f"{key} spatial shape does not match metadata.grid_size: {path}")
    return payload


def build_track_feature_index(
    *,
    feature_index_path: Path,
    track_cache_path: Path,
    output_feature_index_path: Path,
    track_feature_root: Path | None = None,
    grid_size: tuple[int, int] | None = None,
    z_patch_size: int = 2,
    trajectory_mode: str = TRAJECTORY_MODE_CAUSAL,
    limit_samples: int | None = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Write per-sample track tensors and an augmented feature index."""

    if output_feature_index_path.resolve() == feature_index_path.resolve():
        raise ValueError("output_feature_index_path must not overwrite the input feature index")
    if output_feature_index_path.exists() and not overwrite:
        raise FileExistsError(
            f"Augmented feature index already exists: {output_feature_index_path}"
        )
    if z_patch_size <= 0:
        raise ValueError("z_patch_size must be positive")
    records = read_jsonl(feature_index_path)
    if limit_samples is not None:
        if limit_samples <= 0:
            raise ValueError("limit_samples must be positive")
        records = records[:limit_samples]
    if not records:
        raise ValueError(f"No feature records available from {feature_index_path}")

    active_grid_size = grid_size or infer_grid_size_from_feature_record(records[0], z_patch_size)
    objects_by_frame = load_raw_track_cache(track_cache_path)
    root = track_feature_root or default_track_feature_root(output_feature_index_path)
    augmented_records = []
    for record in records:
        payload = rasterize_sample_grids(
            record,
            objects_by_frame,
            grid_size=active_grid_size,
            trajectory_mode=trajectory_mode,
        )
        sample_id = str(record["sample_id"])
        split = str(record.get("split", "samples"))
        track_path = root / split / f"{sample_id}.pt"
        if track_path.exists() and not overwrite:
            raise FileExistsError(f"Track feature tensor already exists: {track_path}")
        save_track_feature_payload(
            track_path,
            {
                "sample_id": sample_id,
                "video_id": record["video_id"],
                "context_grid": payload["context_grid"],
                "future_grid": payload["future_grid"],
                "metadata": payload["metadata"],
            },
        )
        augmented = dict(record)
        augmented["track_feature_path"] = str(track_path)
        augmented_records.append(augmented)
    write_jsonl(output_feature_index_path, augmented_records)
    return augmented_records


def infer_grid_size_from_feature_record(
    record: dict[str, Any], z_patch_size: int
) -> tuple[int, int]:
    """Infer the DiT patch grid from one cached Z tensor record."""

    if "z_path" not in record:
        raise ValueError("feature records must contain z_path when grid_size is not provided")
    payload = torch.load(Path(record["z_path"]), map_location="cpu", weights_only=False)
    z = payload.get("z")
    if not isinstance(z, torch.Tensor) or z.ndim != 4:
        raise ValueError("cached z tensor must be shaped [T, C, H, W]")
    height, width = int(z.shape[-2]), int(z.shape[-1])
    if height % z_patch_size or width % z_patch_size:
        raise ValueError("z_patch_size must divide cached Z height and width")
    return (height // z_patch_size, width // z_patch_size)


def default_track_feature_root(output_feature_index_path: Path) -> Path:
    """Return the default root for generated track feature tensors."""

    return output_feature_index_path.parent / "track" / output_feature_index_path.stem


def write_track_cache_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write raw track-cache JSONL records; useful for smoke tests and fixtures."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for track feature generation."""

    parser = argparse.ArgumentParser(description="Build dense track feature grids.")
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--track-cache", type=Path, required=True)
    parser.add_argument("--output-feature-index", type=Path, required=True)
    parser.add_argument("--track-feature-root", type=Path, default=None)
    parser.add_argument("--grid-height", type=int, default=None)
    parser.add_argument("--grid-width", type=int, default=None)
    parser.add_argument("--z-patch-size", type=int, default=2)
    parser.add_argument(
        "--trajectory-mode",
        choices=sorted(TRAJECTORY_MODES),
        default=TRAJECTORY_MODE_CAUSAL,
        help="causal uses history up to each frame; window uses the full sampled window offline.",
    )
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""

    args = parse_args(argv)
    if (args.grid_height is None) != (args.grid_width is None):
        raise ValueError("--grid-height and --grid-width must be provided together")
    grid_size = (
        (args.grid_height, args.grid_width)
        if args.grid_height is not None and args.grid_width is not None
        else None
    )
    records = build_track_feature_index(
        feature_index_path=args.feature_index,
        track_cache_path=args.track_cache,
        output_feature_index_path=args.output_feature_index,
        track_feature_root=args.track_feature_root,
        grid_size=grid_size,
        z_patch_size=args.z_patch_size,
        trajectory_mode=args.trajectory_mode,
        limit_samples=args.limit_samples,
        overwrite=args.overwrite,
    )
    print(f"wrote {len(records)} augmented feature records -> {args.output_feature_index}")


if __name__ == "__main__":
    main(sys.argv[1:])
