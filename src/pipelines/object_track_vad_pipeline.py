"""Object/track-only anomaly scoring without DiT features."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from src.pipelines.feature_eng_pipeline import read_jsonl, scene_class, write_jsonl
from src.pipelines.object_track_pipeline import (
    TrackFrame,
    TrackObject,
    collect_jsonl_paths,
    fill_missing_velocities,
    parse_track_frame,
)
from src.pipelines.z_dit_pipeline import compute_binary_metrics, compute_score_metrics, write_json

STATE_FIELDS = (
    "cx",
    "cy",
    "w",
    "h",
    "vx",
    "vy",
    "speed",
    "ax",
    "ay",
    "class_id",
    "confidence",
    "age",
    "missing_count",
)
STATE_DIM = len(STATE_FIELDS)
STATE_ZERO = tuple(0.0 for _ in STATE_FIELDS)
GLOBAL_GROUP = "__global__"
SCORE_NEAREST_MEMORY = "nearest_memory_distance"
SCORE_TRAJECTORY_VELOCITY = "trajectory_velocity_distance"
SCORE_CLASS_SCENE_MEMORY = "class_scene_memory_distance"
SCORE_TOPK_TRACK_FRAME = "topk_track_frame_score"
SCORE_MODES = (
    SCORE_NEAREST_MEMORY,
    SCORE_TRAJECTORY_VELOCITY,
    SCORE_CLASS_SCENE_MEMORY,
    SCORE_TOPK_TRACK_FRAME,
)
FRAME_AGGREGATIONS = ("max", "topk_mean", "mean")
PROTOTYPE_METHODS = ("random", "kmeans")
VELOCITY_FIELD_NAMES = ("vx", "vy", "speed", "ax", "ay")
VELOCITY_FIELD_INDICES = tuple(STATE_FIELDS.index(name) for name in VELOCITY_FIELD_NAMES)


@dataclass(frozen=True)
class TrackStateObservation:
    """One object/track observation converted to a normalized state vector."""

    video_id: str
    scene_id: str | None
    frame_idx: int
    track_id: str
    class_id: int | None
    class_name: str | None
    confidence: float
    state: tuple[float, ...]


@dataclass(frozen=True)
class TrackWindow:
    """A fixed-length per-track temporal window ending at one observed frame."""

    video_id: str
    scene_id: str | None
    frame_idx: int
    track_id: str
    class_id: int | None
    class_name: str | None
    states: tuple[tuple[float, ...], ...]
    mask: tuple[bool, ...]

    def vector(self) -> torch.Tensor:
        """Return flattened states plus a binary observation mask."""

        state_values = [value for state in self.states for value in state]
        mask_values = [1.0 if value else 0.0 for value in self.mask]
        return torch.tensor(state_values + mask_values, dtype=torch.float32)


@dataclass(frozen=True)
class MemoryBank:
    """Prototype memory built from train-normal object/track windows."""

    window_length: int
    state_fields: tuple[str, ...]
    prototype_method: str
    groups: dict[str, torch.Tensor]
    counts: dict[str, int]
    max_prototypes: int


def evaluate_object_track_vad(
    *,
    train_track_cache: Path,
    test_track_cache: Path,
    output_dir: Path,
    labels_path: Path | None = None,
    test_feature_index: Path | None = None,
    train_labels_path: Path | None = None,
    context_length: int = 8,
    max_prototypes: int = 2048,
    prototype_method: str = "random",
    primary_score_mode: str = SCORE_CLASS_SCENE_MEMORY,
    frame_aggregation: str = "topk_mean",
    frame_top_k: int = 3,
    seed: int = 0,
    overwrite: bool = False,
) -> Path:
    """Run the object-track-only baseline and write scores/metrics."""

    if output_dir.exists() and not overwrite and (output_dir / "metrics.json").exists():
        raise FileExistsError(f"Object-track VAD output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_options(
        context_length=context_length,
        max_prototypes=max_prototypes,
        prototype_method=prototype_method,
        primary_score_mode=primary_score_mode,
        frame_aggregation=frame_aggregation,
        frame_top_k=frame_top_k,
    )

    train_labels = load_labels(train_labels_path, None) if train_labels_path else {}
    train_cache = load_track_state_cache(train_track_cache)
    train_windows = build_track_windows(
        train_cache.observations,
        context_length=context_length,
        labels=train_labels,
        normal_only=bool(train_labels),
    )
    memory_bank = build_memory_bank(
        train_windows,
        context_length=context_length,
        max_prototypes=max_prototypes,
        prototype_method=prototype_method,
        seed=seed,
    )
    memory_path = output_dir / "memory_bank.pt"
    save_memory_bank(memory_bank, memory_path)

    labels = load_labels(labels_path, test_feature_index)
    test_cache = load_track_state_cache(test_track_cache)
    test_windows = build_track_windows(test_cache.observations, context_length=context_length)
    track_scores = score_track_windows(test_windows, memory_bank)
    frame_scores = aggregate_frame_scores_from_tracks(
        frames=test_cache.frames,
        track_score_records=track_scores,
        labels=labels,
        primary_score_mode=primary_score_mode,
        frame_aggregation=frame_aggregation,
        frame_top_k=frame_top_k,
    )

    write_jsonl(output_dir / "frame_scores.jsonl", frame_scores)
    write_track_debug_csv(output_dir / "per_track_scores.csv", track_scores)
    metrics = compute_binary_metrics(frame_scores)
    payload: dict[str, Any] = {
        **metrics,
        "primary_score_mode": primary_score_mode,
        "frame_aggregation": frame_aggregation,
        "frame_top_k": frame_top_k,
        "context_length": context_length,
        "prototype_method": prototype_method,
        "max_prototypes": max_prototypes,
        "memory_bank_path": str(memory_path),
        "num_train_windows": len(train_windows),
        "num_test_windows": len(test_windows),
        "num_frame_scores": len(frame_scores),
        "score_metrics": compute_score_metrics(frame_scores),
    }
    write_json(output_dir / "metrics.json", payload)
    return output_dir


@dataclass(frozen=True)
class TrackStateCache:
    """Loaded track cache with frame metadata and normalized observations."""

    frames: tuple[TrackFrame, ...]
    observations: tuple[TrackStateObservation, ...]


def validate_options(
    *,
    context_length: int,
    max_prototypes: int,
    prototype_method: str,
    primary_score_mode: str,
    frame_aggregation: str,
    frame_top_k: int,
) -> None:
    """Validate user-facing scoring options."""

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if max_prototypes <= 0:
        raise ValueError("max_prototypes must be positive")
    if prototype_method not in PROTOTYPE_METHODS:
        raise ValueError(f"prototype_method must be one of {PROTOTYPE_METHODS}")
    if primary_score_mode not in SCORE_MODES:
        raise ValueError(f"primary_score_mode must be one of {SCORE_MODES}")
    if frame_aggregation not in FRAME_AGGREGATIONS:
        raise ValueError(f"frame_aggregation must be one of {FRAME_AGGREGATIONS}")
    if frame_top_k <= 0:
        raise ValueError("frame_top_k must be positive")


def load_track_state_cache(path: Path) -> TrackStateCache:
    """Load raw detector/tracker JSONL and convert objects to state vectors."""

    frames: list[TrackFrame] = []
    scene_by_frame: dict[tuple[str, int], str | None] = {}
    missing_by_object: dict[tuple[str, int, str], int] = {}
    all_objects: list[TrackObject] = []
    for jsonl_path in collect_jsonl_paths(path):
        for line_number, raw in enumerate(read_jsonl(jsonl_path), start=1):
            try:
                frame = parse_track_frame(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid track cache record {jsonl_path}:{line_number}: {error}"
                ) from error
            frames.append(frame)
            scene_by_frame[(frame.video_id, frame.frame_idx)] = parse_scene_id(raw)
            objects_value = raw.get("objects", [])
            for obj, raw_obj in zip(frame.objects, objects_value, strict=True):
                missing_by_object[(obj.video_id, obj.frame_idx, obj.track_id)] = parse_missing_count(
                    raw_obj
                )
            all_objects.extend(frame.objects)

    filled_objects = fill_missing_velocities(all_objects)
    acceleration_by_object = compute_accelerations(filled_objects)
    observations = [
        object_to_state_observation(
            obj,
            scene_id=scene_by_frame.get((obj.video_id, obj.frame_idx)),
            missing_count=missing_by_object.get((obj.video_id, obj.frame_idx, obj.track_id), 0),
            acceleration=acceleration_by_object[(obj.video_id, obj.frame_idx, obj.track_id)],
        )
        for obj in sorted(filled_objects, key=lambda item: (item.video_id, item.frame_idx, item.track_id))
    ]
    return TrackStateCache(
        frames=tuple(sorted(frames, key=lambda item: (item.video_id, item.frame_idx))),
        observations=tuple(observations),
    )


def parse_scene_id(raw: dict[str, Any]) -> str | None:
    """Return scene id from raw metadata or infer it from the video id."""

    value = raw.get("scene_id")
    if isinstance(value, str) and value:
        return value
    video_id = raw.get("video_id")
    if isinstance(video_id, str) and "_" in video_id:
        return scene_class(video_id)
    return None


def parse_missing_count(raw_obj: Any) -> int:
    """Read tracker missing count metadata when present."""

    if not isinstance(raw_obj, dict):
        return 0
    value = raw_obj.get("missing", raw_obj.get("missing_count", 0))
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def compute_accelerations(objects: list[TrackObject]) -> dict[tuple[str, int, str], tuple[float, float]]:
    """Compute normalized velocity deltas per same-video same-track sequence."""

    grouped: dict[tuple[str, str], list[TrackObject]] = {}
    for obj in objects:
        grouped.setdefault((obj.video_id, obj.track_id), []).append(obj)

    accelerations: dict[tuple[str, int, str], tuple[float, float]] = {}
    for grouped_objects in grouped.values():
        previous_velocity: tuple[float, float] | None = None
        for obj in sorted(grouped_objects, key=lambda item: item.frame_idx):
            vx, vy = obj.velocity
            if previous_velocity is None:
                ax, ay = 0.0, 0.0
            else:
                ax = vx - previous_velocity[0]
                ay = vy - previous_velocity[1]
            accelerations[(obj.video_id, obj.frame_idx, obj.track_id)] = (
                clamp(ax, -1.0, 1.0),
                clamp(ay, -1.0, 1.0),
            )
            previous_velocity = (vx, vy)
    return accelerations


def object_to_state_observation(
    obj: TrackObject,
    *,
    scene_id: str | None,
    missing_count: int,
    acceleration: tuple[float, float],
) -> TrackStateObservation:
    """Convert one validated object to the required normalized state vector."""

    x1, y1, x2, y2 = obj.bbox_xyxy_norm
    cx, cy = obj.center_xy
    vx, vy = obj.velocity
    vx = clamp(vx, -1.0, 1.0)
    vy = clamp(vy, -1.0, 1.0)
    ax, ay = acceleration
    speed = min(math.sqrt(vx * vx + vy * vy), 1.0)
    class_id = obj.class_id
    state = (
        clamp(cx, 0.0, 1.0),
        clamp(cy, 0.0, 1.0),
        clamp(x2 - x1, 0.0, 1.0),
        clamp(y2 - y1, 0.0, 1.0),
        vx,
        vy,
        speed,
        ax,
        ay,
        normalize_class_id(class_id),
        clamp(float(obj.score), 0.0, 1.0),
        normalize_count(obj.age or 0, scale=100.0),
        normalize_count(missing_count, scale=20.0),
    )
    return TrackStateObservation(
        video_id=obj.video_id,
        scene_id=scene_id,
        frame_idx=obj.frame_idx,
        track_id=obj.track_id,
        class_id=class_id,
        class_name=obj.class_name,
        confidence=obj.score,
        state=state,
    )


def normalize_class_id(class_id: int | None) -> float:
    """Map detector class ids to a bounded numeric feature."""

    if class_id is None or class_id < 0:
        return 0.0
    return clamp(float(class_id) / 100.0, 0.0, 1.0)


def normalize_count(value: int, *, scale: float) -> float:
    """Clip and normalize non-spatial count metadata."""

    return clamp(float(max(0, value)) / scale, 0.0, 1.0)


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a float to a closed interval."""

    return max(lower, min(upper, float(value)))


def build_track_windows(
    observations: Iterable[TrackStateObservation],
    *,
    context_length: int,
    labels: dict[tuple[str, int], int] | None = None,
    normal_only: bool = False,
) -> list[TrackWindow]:
    """Build fixed-length causal per-track windows with padding masks."""

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    labels = labels or {}
    grouped: dict[tuple[str, str], dict[int, TrackStateObservation]] = {}
    for observation in observations:
        grouped.setdefault((observation.video_id, observation.track_id), {})[
            observation.frame_idx
        ] = observation

    windows: list[TrackWindow] = []
    for frame_map in grouped.values():
        for frame_idx in sorted(frame_map):
            if normal_only and int(labels.get((frame_map[frame_idx].video_id, frame_idx), 0)) != 0:
                continue
            current = frame_map[frame_idx]
            states = []
            mask = []
            for window_frame in range(frame_idx - context_length + 1, frame_idx + 1):
                observation = frame_map.get(window_frame)
                if observation is None:
                    states.append(STATE_ZERO)
                    mask.append(False)
                else:
                    states.append(observation.state)
                    mask.append(True)
            windows.append(
                TrackWindow(
                    video_id=current.video_id,
                    scene_id=current.scene_id,
                    frame_idx=current.frame_idx,
                    track_id=current.track_id,
                    class_id=current.class_id,
                    class_name=current.class_name,
                    states=tuple(states),
                    mask=tuple(mask),
                )
            )
    return sorted(windows, key=lambda item: (item.video_id, item.frame_idx, item.track_id))


def build_memory_bank(
    windows: list[TrackWindow],
    *,
    context_length: int,
    max_prototypes: int,
    prototype_method: str = "random",
    seed: int = 0,
) -> MemoryBank:
    """Build global, scene, class, and scene-class prototype banks."""

    if not windows:
        raise ValueError("Cannot build memory bank without train-normal track windows")
    validate_options(
        context_length=context_length,
        max_prototypes=max_prototypes,
        prototype_method=prototype_method,
        primary_score_mode=SCORE_CLASS_SCENE_MEMORY,
        frame_aggregation="topk_mean",
        frame_top_k=1,
    )

    grouped_vectors: dict[str, list[torch.Tensor]] = {}
    for window in windows:
        vector = window.vector()
        for group_key in memory_group_keys(window):
            grouped_vectors.setdefault(group_key, []).append(vector)

    groups = {
        key: make_prototypes(
            torch.stack(vectors),
            max_prototypes=max_prototypes,
            prototype_method=prototype_method,
            seed=stable_seed(seed, key),
        )
        for key, vectors in grouped_vectors.items()
        if vectors
    }
    counts = {key: len(vectors) for key, vectors in grouped_vectors.items()}
    return MemoryBank(
        window_length=context_length,
        state_fields=STATE_FIELDS,
        prototype_method=prototype_method,
        groups=groups,
        counts=counts,
        max_prototypes=max_prototypes,
    )


def memory_group_keys(window: TrackWindow) -> tuple[str, ...]:
    """Return all memory groups this window contributes to."""

    keys = [GLOBAL_GROUP]
    if window.scene_id:
        keys.append(scene_group_key(window.scene_id))
    if window.class_id is not None:
        keys.append(class_group_key(window.class_id))
    if window.scene_id and window.class_id is not None:
        keys.append(scene_class_group_key(window.scene_id, window.class_id))
    return tuple(keys)


def scene_group_key(scene_id: str) -> str:
    """Return memory key for scene-level prototypes."""

    return f"scene:{scene_id}"


def class_group_key(class_id: int) -> str:
    """Return memory key for class-level prototypes."""

    return f"class:{class_id}"


def scene_class_group_key(scene_id: str, class_id: int) -> str:
    """Return memory key for joint scene/class prototypes."""

    return f"scene:{scene_id}|class:{class_id}"


def stable_seed(seed: int, value: str) -> int:
    """Return a deterministic group-specific seed without relying on hash randomization."""

    total = int(seed)
    for char in value:
        total = (total * 33 + ord(char)) % 2_147_483_647
    return total


def make_prototypes(
    vectors: torch.Tensor,
    *,
    max_prototypes: int,
    prototype_method: str,
    seed: int,
) -> torch.Tensor:
    """Subsample or cluster vectors into memory prototypes."""

    if vectors.shape[0] <= max_prototypes:
        return vectors.detach().cpu().clone()
    if prototype_method == "random":
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(vectors.shape[0], generator=generator)[:max_prototypes]
        return vectors.index_select(0, indices).detach().cpu().clone()
    if prototype_method == "kmeans":
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError as error:
            raise RuntimeError("prototype_method='kmeans' requires scikit-learn") from error
        model = MiniBatchKMeans(
            n_clusters=max_prototypes,
            random_state=seed,
            n_init="auto",
            batch_size=max(1024, max_prototypes * 4),
        )
        model.fit(vectors.detach().cpu().numpy())
        return torch.tensor(model.cluster_centers_, dtype=torch.float32)
    raise ValueError(f"Unsupported prototype_method: {prototype_method}")


def save_memory_bank(memory_bank: MemoryBank, path: Path) -> None:
    """Persist a memory bank to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "window_length": memory_bank.window_length,
            "state_fields": list(memory_bank.state_fields),
            "prototype_method": memory_bank.prototype_method,
            "max_prototypes": memory_bank.max_prototypes,
            "counts": memory_bank.counts,
            "groups": memory_bank.groups,
        },
        path,
    )


def load_memory_bank(path: Path) -> MemoryBank:
    """Load a saved memory bank."""

    payload = torch.load(path, map_location="cpu")
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        raise ValueError(f"Invalid memory bank groups: {path}")
    return MemoryBank(
        window_length=int(payload["window_length"]),
        state_fields=tuple(payload["state_fields"]),
        prototype_method=str(payload.get("prototype_method", "unknown")),
        groups={str(key): value.float().cpu() for key, value in groups.items()},
        counts={str(key): int(value) for key, value in payload.get("counts", {}).items()},
        max_prototypes=int(payload.get("max_prototypes", 0)),
    )


def score_track_windows(
    windows: list[TrackWindow],
    memory_bank: MemoryBank,
) -> list[dict[str, Any]]:
    """Score every test track window against memory prototypes."""

    records = []
    for window in windows:
        vector = window.vector()
        global_prototypes = require_memory_group(memory_bank, GLOBAL_GROUP)
        nearest_score = nearest_distance(vector, global_prototypes)
        trajectory_score = nearest_distance(
            vector,
            global_prototypes,
            feature_indices=trajectory_velocity_indices(memory_bank.window_length),
        )
        class_scene_score = nearest_distance(vector, select_group_prototypes(memory_bank, window))
        records.append(
            {
                "video_id": window.video_id,
                "scene_id": window.scene_id,
                "frame_idx": window.frame_idx,
                "track_id": window.track_id,
                "class_id": window.class_id,
                "class_name": window.class_name,
                SCORE_NEAREST_MEMORY: nearest_score,
                SCORE_TRAJECTORY_VELOCITY: trajectory_score,
                SCORE_CLASS_SCENE_MEMORY: class_scene_score,
            }
        )
    return records


def require_memory_group(memory_bank: MemoryBank, group_key: str) -> torch.Tensor:
    """Return a prototype group or fail clearly."""

    prototypes = memory_bank.groups.get(group_key)
    if prototypes is None or prototypes.numel() == 0:
        raise ValueError(f"Missing memory prototypes for group {group_key!r}")
    return prototypes


def select_group_prototypes(memory_bank: MemoryBank, window: TrackWindow) -> torch.Tensor:
    """Select scene+class, class, scene, then global prototypes for one window."""

    candidate_keys: list[str] = []
    if window.scene_id and window.class_id is not None:
        candidate_keys.append(scene_class_group_key(window.scene_id, window.class_id))
    if window.class_id is not None:
        candidate_keys.append(class_group_key(window.class_id))
    if window.scene_id:
        candidate_keys.append(scene_group_key(window.scene_id))
    candidate_keys.append(GLOBAL_GROUP)
    for key in candidate_keys:
        prototypes = memory_bank.groups.get(key)
        if prototypes is not None and prototypes.numel() > 0:
            return prototypes
    raise ValueError("Memory bank has no usable prototypes")


def nearest_distance(
    vector: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    feature_indices: torch.Tensor | None = None,
) -> float:
    """Return nearest mean squared Euclidean distance to prototypes."""

    vector = vector.float().cpu()
    prototypes = prototypes.float().cpu()
    if feature_indices is not None:
        vector = vector.index_select(0, feature_indices)
        prototypes = prototypes.index_select(1, feature_indices)
    diff = prototypes - vector.unsqueeze(0)
    distances = torch.mean(diff * diff, dim=1)
    return float(torch.min(distances).item())


def trajectory_velocity_indices(window_length: int) -> torch.Tensor:
    """Return flattened state indices for velocity/speed/acceleration features."""

    indices = []
    for time_index in range(window_length):
        offset = time_index * STATE_DIM
        for field_index in VELOCITY_FIELD_INDICES:
            indices.append(offset + field_index)
    return torch.tensor(indices, dtype=torch.long)


def aggregate_frame_scores_from_tracks(
    *,
    frames: tuple[TrackFrame, ...],
    track_score_records: list[dict[str, Any]],
    labels: dict[tuple[str, int], int],
    primary_score_mode: str,
    frame_aggregation: str,
    frame_top_k: int,
) -> list[dict[str, Any]]:
    """Aggregate per-track scores into one frame-level JSONL-compatible record."""

    scores_by_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in track_score_records:
        scores_by_frame.setdefault((record["video_id"], int(record["frame_idx"])), []).append(record)

    frame_scene: dict[tuple[str, int], str | None] = {
        (frame.video_id, frame.frame_idx): parse_scene_id(
            {"video_id": frame.video_id, "scene_id": None}
        )
        for frame in frames
    }
    for record in track_score_records:
        key = (record["video_id"], int(record["frame_idx"]))
        if frame_scene.get(key) is None and record.get("scene_id"):
            frame_scene[key] = str(record["scene_id"])

    all_keys = sorted(
        set(frame_scene) | set(scores_by_frame) | set(labels),
        key=lambda item: (item[0], item[1]),
    )
    frame_scores = []
    for video_id, frame_idx in all_keys:
        track_records = scores_by_frame.get((video_id, frame_idx), [])
        variant_scores = {
            SCORE_NEAREST_MEMORY: [float(item[SCORE_NEAREST_MEMORY]) for item in track_records],
            SCORE_TRAJECTORY_VELOCITY: [
                float(item[SCORE_TRAJECTORY_VELOCITY]) for item in track_records
            ],
            SCORE_CLASS_SCENE_MEMORY: [
                float(item[SCORE_CLASS_SCENE_MEMORY]) for item in track_records
            ],
        }
        aggregated = {
            mode: aggregate_values(values, mode=frame_aggregation, top_k=frame_top_k)
            for mode, values in variant_scores.items()
        }
        aggregated[SCORE_TOPK_TRACK_FRAME] = aggregate_values(
            variant_scores[SCORE_NEAREST_MEMORY],
            mode="topk_mean",
            top_k=frame_top_k,
        )
        primary = float(aggregated[primary_score_mode])
        frame_record: dict[str, Any] = {
            "video_id": video_id,
            "scene_id": frame_scene.get((video_id, frame_idx)),
            "frame_idx": frame_idx,
            "label": int(labels.get((video_id, frame_idx), 0)),
            "score": primary,
            "raw_score": primary,
            "num_tracks": len(track_records),
            "num_votes": 1,
        }
        for mode in SCORE_MODES:
            frame_record[f"{mode}_raw_score"] = float(aggregated[mode])
        frame_scores.append(frame_record)
    return frame_scores


def aggregate_values(values: list[float], *, mode: str, top_k: int) -> float:
    """Aggregate per-track values for one frame."""

    if not values:
        return 0.0
    if mode == "max":
        return float(max(values))
    if mode == "mean":
        return float(sum(values) / len(values))
    if mode == "topk_mean":
        ordered = sorted(values, reverse=True)
        selected = ordered[: min(top_k, len(ordered))]
        return float(sum(selected) / len(selected))
    raise ValueError(f"Unsupported frame aggregation: {mode}")


def load_labels(
    labels_path: Path | None,
    feature_index_path: Path | None,
) -> dict[tuple[str, int], int]:
    """Load frame labels from CSV and/or existing feature-index metadata."""

    labels: dict[tuple[str, int], int] = {}
    if feature_index_path is not None:
        for record in read_jsonl(feature_index_path):
            video_id = str(record["video_id"])
            frames = record.get("future_frames", [])
            frame_labels = record.get("future_frame_labels", [])
            for frame_idx, label in zip(frames, frame_labels, strict=True):
                key = (video_id, int(frame_idx))
                labels[key] = max(labels.get(key, 0), int(label))
    if labels_path is not None:
        with labels_path.open("r", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                key = (row["video_id"], int(row["frame_idx"]))
                labels[key] = max(labels.get(key, 0), int(row["label"]))
    return labels


def write_track_debug_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write per-track scores for debugging."""

    fieldnames = [
        "video_id",
        "scene_id",
        "frame_idx",
        "track_id",
        "class_id",
        "class_name",
        SCORE_NEAREST_MEMORY,
        SCORE_TRAJECTORY_VELOCITY,
        SCORE_CLASS_SCENE_MEMORY,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
