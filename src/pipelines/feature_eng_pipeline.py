"""Dataset preprocessing from raw data to processed frame windows."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image

from src.utils import configure_progress_from_config, progress_bar

NATIVE_FPS = "native"
IMAGE_BACKEND_AUTO = "auto"
IMAGE_BACKEND_OPENCV = "opencv"
IMAGE_BACKEND_PILLOW = "pillow"
VIDEO_SOURCE = "video"
FRAME_DIR_SOURCE = "frame_dir"
TRAIN_SPLIT = "train_normal"
TEST_SPLIT = "test"
SHANGHAITECH_DATASETS = {"shanghaitech", "shanghaitech_campus"}
AVENUE_DATASETS = {"avenue", "cuhk_avenue"}
SUPPORTED_DATASETS = SHANGHAITECH_DATASETS | AVENUE_DATASETS


@dataclass(frozen=True)
class SamplingConfig:
    """Context/future window sampling settings."""

    context_frames: int = 32
    future_frames: int = 8
    context_sampling: str = "dense"
    context_span: int = 32
    gap: int = 0
    stride_train: int = 8
    stride_test: int = 4
    window_label: str = "any_future_anomaly"


@dataclass(frozen=True)
class PreprocessConfig:
    """Resolved preprocessing configuration."""

    raw_root: Path
    processed_root: Path
    dataset_name: str = "shanghaitech"
    fps: int | None = None
    image_ext: str = "jpg"
    jpeg_quality: int = 2
    overwrite: bool = False
    frame_index_start: int = 1
    num_workers: int = 8
    image_backend: str = IMAGE_BACKEND_AUTO
    label_source: str = "test_frame_mask"
    label_mismatch_policy: str = "error"
    sampling: SamplingConfig = SamplingConfig()

    @property
    def frames_root(self) -> Path:
        return self.processed_root / "frames"

    @property
    def manifest_path(self) -> Path:
        return self.processed_root / "video_manifest.csv"

    @property
    def frame_index_map_path(self) -> Path:
        return self.processed_root / "frame_index_map.csv"

    @property
    def frame_labels_path(self) -> Path:
        return self.processed_root / "frame_labels.csv"

    @property
    def train_samples_path(self) -> Path:
        return self.processed_root / "samples_train.jsonl"

    @property
    def test_samples_path(self) -> Path:
        return self.processed_root / "samples_test.jsonl"


@dataclass(frozen=True)
class VideoRecord:
    """A raw video or frame sequence selected for preprocessing."""

    video_id: str
    split: str
    class_name: str
    is_anomaly: bool
    source_path: Path
    frame_dir: Path
    source_type: str
    mask_path: Path | None = None
    source_fps: float | None = None


@dataclass(frozen=True)
class FrameMap:
    """Mapping from a raw source frame to a processed frame."""

    processed_frame_idx: int
    processed_frame_path: Path
    raw_frame_idx: int | None
    raw_frame_path: Path | None
    timestamp_sec: float | None
    source_position: int | None = None


@dataclass(frozen=True)
class PreparedVideo:
    """One processed video sequence and its raw-to-processed frame map."""

    num_frames: int
    effective_fps: float | None
    frame_maps: list[FrameMap]


def load_config(path: Path) -> PreprocessConfig:
    """Load and validate preprocessing settings from YAML."""

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    configure_progress_from_config(config)
    dataset = config.get("dataset", {})
    dataset_name = str(dataset.get("name", "")).lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            "Expected dataset.name to be one of: "
            f"{', '.join(sorted(SUPPORTED_DATASETS))}. Got: {dataset_name!r}"
        )

    preprocess = config.get("preprocess", {})
    sampling = load_sampling_config(config.get("sampling", {}))
    image_ext = str(preprocess.get("image_ext", "jpg")).lower().lstrip(".")
    if image_ext not in {"jpg", "jpeg", "png"}:
        raise ValueError("preprocess.image_ext must be one of: jpg, jpeg, png")

    default_label_source = (
        "test_frame_mask"
        if dataset_name in SHANGHAITECH_DATASETS
        else "ground_truth_demo/testing_label_mask"
    )
    label_source = str(preprocess.get("label_source", default_label_source))
    validate_label_source(dataset_name, label_source)

    mismatch_policy = str(preprocess.get("label_mismatch_policy", "error"))
    if mismatch_policy != "error":
        raise ValueError("Only preprocess.label_mismatch_policy='error' is supported")

    frame_index_start = int(preprocess.get("frame_index_start", 1))
    if frame_index_start <= 0:
        raise ValueError("preprocess.frame_index_start must be positive")

    num_workers = int(preprocess.get("num_workers", 8))
    if num_workers < 0:
        raise ValueError("preprocess.num_workers must be non-negative")

    image_backend = str(preprocess.get("image_backend", IMAGE_BACKEND_AUTO)).lower()
    if image_backend not in {
        IMAGE_BACKEND_AUTO,
        IMAGE_BACKEND_OPENCV,
        IMAGE_BACKEND_PILLOW,
    }:
        raise ValueError("preprocess.image_backend must be 'auto', 'opencv', or 'pillow'")

    return PreprocessConfig(
        dataset_name=dataset_name,
        raw_root=Path(dataset["raw_root"]),
        processed_root=Path(dataset["processed_root"]),
        fps=parse_fps(preprocess.get("fps", NATIVE_FPS)),
        image_ext=image_ext,
        jpeg_quality=int(preprocess.get("jpeg_quality", 2)),
        overwrite=bool(preprocess.get("overwrite", False)),
        frame_index_start=frame_index_start,
        num_workers=num_workers,
        image_backend=image_backend,
        label_source=label_source,
        label_mismatch_policy=mismatch_policy,
        sampling=sampling,
    )


def validate_label_source(dataset_name: str, label_source: str) -> None:
    """Validate the dataset-specific label source setting."""

    if dataset_name in SHANGHAITECH_DATASETS and label_source != "test_frame_mask":
        raise ValueError("ShanghaiTech requires preprocess.label_source='test_frame_mask'")
    if dataset_name in AVENUE_DATASETS and not label_source:
        raise ValueError("Avenue requires a non-empty preprocess.label_source")


def load_sampling_config(config: dict[str, Any]) -> SamplingConfig:
    """Load and validate context/future window settings."""

    sampling = SamplingConfig(
        context_frames=int(config.get("context_frames", 32)),
        future_frames=int(config.get("future_frames", 8)),
        context_sampling=str(config.get("context_sampling", "dense")),
        context_span=int(config.get("context_span", config.get("context_frames", 32))),
        gap=int(config.get("gap", 0)),
        stride_train=int(config.get("stride_train", config.get("stride", 8))),
        stride_test=int(config.get("stride_test", config.get("stride", 4))),
        window_label=str(config.get("window_label", "any_future_anomaly")),
    )
    validate_sampling_config(sampling)
    return sampling


def validate_sampling_config(config: SamplingConfig) -> None:
    """Validate window sampling settings."""

    positive_fields = {
        "context_frames": config.context_frames,
        "future_frames": config.future_frames,
        "stride_train": config.stride_train,
        "stride_test": config.stride_test,
    }
    for name, value in positive_fields.items():
        if value <= 0:
            raise ValueError(f"sampling.{name} must be positive")
    if config.gap < 0:
        raise ValueError("sampling.gap must be non-negative")
    if config.context_sampling not in {"dense", "sparse_uniform"}:
        raise ValueError("sampling.context_sampling must be 'dense' or 'sparse_uniform'")
    if config.context_sampling == "dense" and config.context_span != config.context_frames:
        raise ValueError("sampling.context_span must equal context_frames for dense sampling")
    if config.context_span < config.context_frames:
        raise ValueError("sampling.context_span must be >= sampling.context_frames")
    if config.window_label != "any_future_anomaly":
        raise ValueError("Only sampling.window_label='any_future_anomaly' is supported")


def parse_fps(value: Any) -> int | None:
    """Parse ``native`` or a positive integer FPS value."""

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == NATIVE_FPS:
            return None
        if normalized.isdigit():
            fps = int(normalized)
        else:
            raise ValueError("preprocess.fps must be 'native' or a positive integer")
    elif isinstance(value, int):
        fps = value
    else:
        raise ValueError("preprocess.fps must be 'native' or a positive integer")

    if fps <= 0:
        raise ValueError("preprocess.fps must be 'native' or a positive integer")
    return fps


def scene_class(video_id: str) -> str:
    """Return a stable scene label from a ShanghaiTech video id."""

    return f"scene_{video_id.split('_', maxsplit=1)[0]}"


def build_video_records(config: PreprocessConfig) -> list[VideoRecord]:
    """Build train-normal and test records from raw data."""

    if config.dataset_name in SHANGHAITECH_DATASETS:
        return build_shanghaitech_video_records(config)
    if config.dataset_name in AVENUE_DATASETS:
        return build_avenue_video_records(config)
    raise ValueError(f"Unsupported dataset.name: {config.dataset_name!r}")


def build_shanghaitech_video_records(config: PreprocessConfig) -> list[VideoRecord]:
    """Build ShanghaiTech train-normal and test records from raw data."""

    train_video_root = config.raw_root / "training" / "videos"
    test_frame_root = config.raw_root / "testing" / "frames"
    test_frame_mask_root = config.raw_root / "testing" / config.label_source

    if not train_video_root.is_dir():
        raise FileNotFoundError(f"Missing ShanghaiTech train videos: {train_video_root}")
    if not test_frame_root.is_dir():
        raise FileNotFoundError(f"Missing ShanghaiTech test frames: {test_frame_root}")
    if not test_frame_mask_root.is_dir():
        raise FileNotFoundError(f"Missing ShanghaiTech test masks: {test_frame_mask_root}")

    records: list[VideoRecord] = []
    for video_path in sorted(train_video_root.glob("*.avi")):
        video_id = video_path.stem
        records.append(
            VideoRecord(
                video_id=video_id,
                split=TRAIN_SPLIT,
                class_name=scene_class(video_id),
                is_anomaly=False,
                source_path=video_path,
                frame_dir=config.frames_root / TRAIN_SPLIT / video_id,
                source_type=VIDEO_SOURCE,
                source_fps=probe_video_fps(video_path),
            )
        )

    for frame_source_dir in sorted(test_frame_root.iterdir()):
        if not frame_source_dir.is_dir():
            continue
        video_id = frame_source_dir.name
        mask_path = test_frame_mask_root / f"{video_id}.npy"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing frame mask: {mask_path}")
        labels = load_frame_mask(mask_path)
        records.append(
            VideoRecord(
                video_id=video_id,
                split=TEST_SPLIT,
                class_name=scene_class(video_id),
                is_anomaly=bool(labels.any()),
                source_path=frame_source_dir,
                frame_dir=config.frames_root / TEST_SPLIT / video_id,
                source_type=FRAME_DIR_SOURCE,
                mask_path=mask_path,
            )
        )
    return records


def build_avenue_video_records(config: PreprocessConfig) -> list[VideoRecord]:
    """Build CUHK Avenue train-normal and test records from raw data."""

    train_video_root = config.raw_root / "training_videos"
    test_video_root = config.raw_root / "testing_videos"
    label_root = config.raw_root / config.label_source

    if not train_video_root.is_dir():
        raise FileNotFoundError(f"Missing Avenue train videos: {train_video_root}")
    if not test_video_root.is_dir():
        raise FileNotFoundError(f"Missing Avenue test videos: {test_video_root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"Missing Avenue test labels: {label_root}")

    records: list[VideoRecord] = []
    for video_path in sorted(train_video_root.glob("*.avi"), key=video_sort_key):
        video_id = video_path.stem
        records.append(
            VideoRecord(
                video_id=video_id,
                split=TRAIN_SPLIT,
                class_name="avenue",
                is_anomaly=False,
                source_path=video_path,
                frame_dir=config.frames_root / TRAIN_SPLIT / video_id,
                source_type=VIDEO_SOURCE,
                source_fps=probe_video_fps(video_path),
            )
        )

    for video_path in sorted(test_video_root.glob("*.avi"), key=video_sort_key):
        video_id = video_path.stem
        mask_path = avenue_label_path(label_root, video_id)
        labels = load_frame_mask(mask_path)
        records.append(
            VideoRecord(
                video_id=video_id,
                split=TEST_SPLIT,
                class_name="avenue",
                is_anomaly=bool(labels.any()),
                source_path=video_path,
                frame_dir=config.frames_root / TEST_SPLIT / video_id,
                source_type=VIDEO_SOURCE,
                mask_path=mask_path,
                source_fps=probe_video_fps(video_path),
            )
        )
    return records


def video_sort_key(path: Path) -> tuple[int, int | str]:
    """Sort numeric video names before non-numeric names."""

    if path.stem.isdigit():
        return (0, int(path.stem))
    return (1, path.stem)


def avenue_label_path(label_root: Path, video_id: str) -> Path:
    """Return the Avenue label path for a zero-padded or plain numeric video id."""

    candidates = []
    if video_id.isdigit():
        candidates.append(label_root / f"{int(video_id)}_label.mat")
    candidates.append(label_root / f"{video_id}_label.mat")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing Avenue label for video {video_id!r}. Tried: "
        f"{', '.join(str(candidate) for candidate in candidates)}"
    )


def select_video_records(
    records: list[VideoRecord],
    *,
    splits: set[str] | None = None,
    limit_videos: int | None = None,
    limit_per_split: int | None = None,
) -> list[VideoRecord]:
    """Select records for full preprocessing or smoke runs."""

    selected = [record for record in records if splits is None or record.split in splits]
    if limit_per_split is not None:
        if limit_per_split <= 0:
            raise ValueError("limit_per_split must be positive")
        counts: dict[str, int] = {}
        limited = []
        for record in selected:
            count = counts.get(record.split, 0)
            if count >= limit_per_split:
                continue
            limited.append(record)
            counts[record.split] = count + 1
        selected = limited
    if limit_videos is not None:
        if limit_videos <= 0:
            raise ValueError("limit_videos must be positive")
        selected = selected[:limit_videos]
    return selected


def run_preprocess(
    config: PreprocessConfig,
    *,
    limit_videos: int | None = None,
    limit_per_split: int | None = None,
    splits: set[str] | None = None,
    overwrite: bool | None = None,
) -> None:
    """Run preprocessing and write all 02_processed artifacts."""

    effective_config = config if overwrite is None else replace(config, overwrite=overwrite)
    records = select_video_records(
        build_video_records(effective_config),
        splits=splits,
        limit_videos=limit_videos,
        limit_per_split=limit_per_split,
    )
    if not records:
        raise ValueError(f"No {effective_config.dataset_name} records selected for preprocessing")
    if any(record.source_type == VIDEO_SOURCE for record in records):
        ensure_ffmpeg_available()

    effective_config.processed_root.mkdir(parents=True, exist_ok=True)
    skipped_short = 0

    with effective_config.manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        with effective_config.frame_index_map_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as map_file:
            with effective_config.frame_labels_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as label_file:
                manifest_writer = csv.DictWriter(
                    manifest_file,
                    fieldnames=[
                        "video_id",
                        "split",
                        "class_name",
                        "is_anomaly",
                        "source_path",
                        "frame_dir",
                        "num_frames",
                        "effective_fps",
                    ],
                )
                map_writer = csv.DictWriter(
                    map_file,
                    fieldnames=[
                        "video_id",
                        "split",
                        "processed_frame_idx",
                        "processed_frame_path",
                        "raw_frame_idx",
                        "raw_frame_path",
                        "timestamp_sec",
                    ],
                )
                label_writer = csv.DictWriter(
                    label_file,
                    fieldnames=[
                        "video_id",
                        "split",
                        "class_name",
                        "frame_idx",
                        "frame_path",
                        "label",
                    ],
                )
                manifest_writer.writeheader()
                map_writer.writeheader()
                label_writer.writeheader()

                for record in progress_bar(records, desc="preprocess videos", unit="video"):
                    prepared = prepare_record(record, effective_config)
                    manifest_writer.writerow(
                        {
                            "video_id": record.video_id,
                            "split": record.split,
                            "class_name": record.class_name,
                            "is_anomaly": int(record.is_anomaly),
                            "source_path": str(record.source_path),
                            "frame_dir": str(record.frame_dir),
                            "num_frames": prepared.num_frames,
                            "effective_fps": format_optional_float(prepared.effective_fps),
                        }
                    )
                    write_frame_index_map(map_writer, record, prepared)
                    write_frame_labels(label_writer, record, prepared, effective_config)
                    if not has_any_window(prepared.num_frames, effective_config.sampling):
                        skipped_short += 1

    train_count, test_count = build_sample_indices(effective_config)
    print(f"wrote manifest -> {effective_config.manifest_path}")
    print(f"wrote frame index map -> {effective_config.frame_index_map_path}")
    print(f"wrote frame labels -> {effective_config.frame_labels_path}")
    print(f"wrote {train_count} train samples -> {effective_config.train_samples_path}")
    print(f"wrote {test_count} test samples -> {effective_config.test_samples_path}")
    if skipped_short:
        print(f"skipped {skipped_short} short videos while building windows")


def prepare_record(record: VideoRecord, config: PreprocessConfig) -> PreparedVideo:
    """Prepare frames for one raw source."""

    if record.source_type == VIDEO_SOURCE:
        return extract_video_frames(record, config)
    if record.source_type == FRAME_DIR_SOURCE:
        return materialize_frame_dir(record, config)
    raise ValueError(f"Unsupported source type: {record.source_type}")


def ensure_ffmpeg_available() -> None:
    """Fail early when ffmpeg is unavailable."""

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for raw video frame extraction")


def probe_video_fps(path: Path) -> float | None:
    """Return video FPS from ffprobe when available."""

    if shutil.which("ffprobe") is None:
        return None
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    value = result.stdout.strip()
    if result.returncode != 0 or not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        denominator_float = float(denominator)
        if denominator_float == 0:
            return None
        return float(numerator) / denominator_float
    return float(value)


def extract_video_frames(record: VideoRecord, config: PreprocessConfig) -> PreparedVideo:
    """Extract frames from a raw video source."""

    existing_count = count_existing_frames(record.frame_dir, config.image_ext)
    if existing_count > 0 and not config.overwrite:
        effective_fps = float(config.fps) if config.fps is not None else record.source_fps
        return PreparedVideo(
            num_frames=existing_count,
            effective_fps=effective_fps,
            frame_maps=build_video_frame_maps(record, config, existing_count, effective_fps),
        )

    record.frame_dir.mkdir(parents=True, exist_ok=True)
    if config.overwrite:
        remove_existing_frames(record.frame_dir, config.image_ext)

    frame_pattern = record.frame_dir / f"%06d.{config.image_ext}"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(record.source_path),
    ]
    if config.fps is not None:
        command.extend(["-vf", f"fps={config.fps}"])
    command.extend(
        [
            "-q:v",
            str(config.jpeg_quality),
            "-start_number",
            str(config.frame_index_start),
            str(frame_pattern),
        ]
    )
    subprocess.run(command, check=True)

    num_frames = count_existing_frames(record.frame_dir, config.image_ext)
    if num_frames == 0:
        raise RuntimeError(f"No frames were extracted for {record.source_path}")
    effective_fps = float(config.fps) if config.fps is not None else record.source_fps
    return PreparedVideo(
        num_frames=num_frames,
        effective_fps=effective_fps,
        frame_maps=build_video_frame_maps(record, config, num_frames, effective_fps),
    )


def build_video_frame_maps(
    record: VideoRecord,
    config: PreprocessConfig,
    num_frames: int,
    effective_fps: float | None,
) -> list[FrameMap]:
    """Build raw-to-processed frame rows for a video source."""

    maps = []
    for offset in range(num_frames):
        frame_idx = config.frame_index_start + offset
        timestamp = offset / effective_fps if effective_fps else None
        source_position = offset
        if record.source_fps and effective_fps:
            source_position = int(round((offset / effective_fps) * record.source_fps))
        maps.append(
            FrameMap(
                processed_frame_idx=frame_idx,
                processed_frame_path=record.frame_dir / f"{frame_idx:06d}.{config.image_ext}",
                raw_frame_idx=source_position + 1,
                raw_frame_path=record.source_path,
                timestamp_sec=timestamp,
                source_position=source_position,
            )
        )
    return maps


def materialize_frame_dir(record: VideoRecord, config: PreprocessConfig) -> PreparedVideo:
    """Copy a raw frame directory into the processed frame layout."""

    source_frames = list_image_files(record.source_path)
    if not source_frames:
        raise RuntimeError(f"No image frames found in {record.source_path}")
    selected = select_frame_dir_sources(source_frames, record.source_fps, config.fps)
    existing_count = count_existing_frames(record.frame_dir, config.image_ext)
    if existing_count > 0 and not config.overwrite:
        if existing_count != len(selected):
            raise RuntimeError(
                f"Existing processed frames for {record.video_id} have count "
                f"{existing_count}, expected {len(selected)}. Rerun with --overwrite."
            )
        return PreparedVideo(
            num_frames=existing_count,
            effective_fps=float(config.fps) if config.fps is not None else record.source_fps,
            frame_maps=build_frame_dir_maps(record, config, selected),
        )

    record.frame_dir.mkdir(parents=True, exist_ok=True)
    if config.overwrite:
        remove_existing_frames(record.frame_dir, config.image_ext)

    copy_tasks = []
    for offset, (_, source_path) in enumerate(selected):
        frame_idx = config.frame_index_start + offset
        target_path = record.frame_dir / f"{frame_idx:06d}.{config.image_ext}"
        copy_tasks.append((source_path, target_path))
    materialize_frame_files(copy_tasks, config)

    return PreparedVideo(
        num_frames=len(selected),
        effective_fps=float(config.fps) if config.fps is not None else record.source_fps,
        frame_maps=build_frame_dir_maps(record, config, selected),
    )


def list_image_files(frame_dir: Path) -> list[Path]:
    """Return image files sorted by numeric frame stem when possible."""

    valid_suffixes = {".jpg", ".jpeg", ".png"}
    frames = [path for path in frame_dir.iterdir() if path.suffix.lower() in valid_suffixes]
    return sorted(frames, key=frame_sort_key)


def frame_sort_key(path: Path) -> tuple[int, int | str]:
    """Sort numeric frame names before non-numeric names."""

    if path.stem.isdigit():
        return (0, int(path.stem))
    return (1, path.stem)


def select_frame_dir_sources(
    source_frames: list[Path],
    source_fps: float | None,
    output_fps: int | None,
) -> list[tuple[int, Path]]:
    """Select frame-directory sources for native or target FPS output."""

    indexed = list(enumerate(source_frames))
    if output_fps is None:
        return indexed
    if source_fps is None:
        raise ValueError(
            "preprocess.fps can only resample frame-directory sources when the "
            "dataset adapter provides source_fps"
        )
    if output_fps > source_fps:
        raise ValueError("preprocess.fps cannot exceed source FPS for frame directories")

    duration = len(source_frames) / source_fps
    expected_count = int(np.floor(duration * output_fps))
    selected: list[tuple[int, Path]] = []
    seen_positions: set[int] = set()
    for output_index in range(expected_count):
        timestamp = output_index / output_fps
        source_position = min(int(round(timestamp * source_fps)), len(source_frames) - 1)
        if source_position in seen_positions:
            continue
        seen_positions.add(source_position)
        selected.append((source_position, source_frames[source_position]))
    return selected


def build_frame_dir_maps(
    record: VideoRecord,
    config: PreprocessConfig,
    selected: list[tuple[int, Path]],
) -> list[FrameMap]:
    """Build raw-to-processed frame rows for a frame-directory source."""

    maps = []
    effective_fps = float(config.fps) if config.fps is not None else record.source_fps
    for offset, (source_position, source_path) in enumerate(selected):
        frame_idx = config.frame_index_start + offset
        timestamp = source_position / effective_fps if effective_fps else None
        maps.append(
            FrameMap(
                processed_frame_idx=frame_idx,
                processed_frame_path=record.frame_dir / f"{frame_idx:06d}.{config.image_ext}",
                raw_frame_idx=parse_frame_index(source_path),
                raw_frame_path=source_path,
                timestamp_sec=timestamp,
                source_position=source_position,
            )
        )
    return maps


def parse_frame_index(path: Path) -> int | None:
    """Parse a frame number from a raw frame filename."""

    return int(path.stem) if path.stem.isdigit() else None


def materialize_frame_files(
    tasks: list[tuple[Path, Path]],
    config: PreprocessConfig,
) -> None:
    """Copy or convert frame files, optionally in parallel."""

    if config.num_workers <= 1 or len(tasks) <= 1:
        for source_path, target_path in progress_bar(
            tasks,
            desc="copy frames",
            unit="frame",
            leave=False,
        ):
            copy_frame_file(
                source_path,
                target_path,
                config.image_ext,
                jpeg_quality=config.jpeg_quality,
                image_backend=config.image_backend,
            )
        return

    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = [
            executor.submit(
                copy_frame_file,
                source_path,
                target_path,
                config.image_ext,
                jpeg_quality=config.jpeg_quality,
                image_backend=config.image_backend,
            )
            for source_path, target_path in tasks
        ]
        for future in progress_bar(
            futures,
            desc="copy frames",
            unit="frame",
            leave=False,
        ):
            future.result()


def copy_frame_file(
    source_path: Path,
    target_path: Path,
    image_ext: str,
    *,
    jpeg_quality: int = 2,
    image_backend: str = IMAGE_BACKEND_AUTO,
) -> None:
    """Copy one raw frame into a separate processed file."""

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower().lstrip(".") == image_ext.lower():
        shutil.copy2(source_path, target_path)
        return
    if image_backend in {IMAGE_BACKEND_AUTO, IMAGE_BACKEND_OPENCV}:
        if copy_frame_file_with_opencv(
            source_path,
            target_path,
            image_ext,
            jpeg_quality=jpeg_quality,
            required=image_backend == IMAGE_BACKEND_OPENCV,
        ):
            return
    copy_frame_file_with_pillow(
        source_path,
        target_path,
        image_ext,
        jpeg_quality=jpeg_quality,
    )


def copy_frame_file_with_opencv(
    source_path: Path,
    target_path: Path,
    image_ext: str,
    *,
    jpeg_quality: int,
    required: bool,
) -> bool:
    """Convert an image with OpenCV when available."""

    try:
        import cv2
    except ImportError:
        if required:
            raise RuntimeError("OpenCV backend requested but cv2 is not installed")
        return False

    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        if required:
            raise RuntimeError(f"OpenCV failed to read image: {source_path}")
        return False

    params: list[int] = []
    normalized_ext = image_ext.lower()
    if normalized_ext in {"jpg", "jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), ffmpeg_quality_to_jpeg_quality(jpeg_quality)]
    if not cv2.imwrite(str(target_path), image, params):
        if required:
            raise RuntimeError(f"OpenCV failed to write image: {target_path}")
        return False
    return True


def copy_frame_file_with_pillow(
    source_path: Path,
    target_path: Path,
    image_ext: str,
    *,
    jpeg_quality: int,
) -> None:
    """Convert an image with Pillow."""

    with Image.open(source_path) as image:
        converted = image.convert("RGB")
        save_kwargs: dict[str, int] = {}
        if image_ext.lower() in {"jpg", "jpeg"}:
            save_kwargs["quality"] = ffmpeg_quality_to_jpeg_quality(jpeg_quality)
        converted.save(target_path, **save_kwargs)


def ffmpeg_quality_to_jpeg_quality(qscale: int) -> int:
    """Map ffmpeg JPEG qscale-like values to Pillow/OpenCV quality."""

    qscale = max(2, min(int(qscale), 31))
    return max(1, min(100, round(100 - ((qscale - 2) * 95 / 29))))


def remove_existing_frames(frame_dir: Path, image_ext: str) -> None:
    """Remove existing processed image files for one sequence."""

    if not frame_dir.is_dir():
        return
    for frame_path in frame_dir.glob(f"*.{image_ext}"):
        frame_path.unlink()


def count_existing_frames(frame_dir: Path, image_ext: str) -> int:
    """Count processed frame files."""

    if not frame_dir.is_dir():
        return 0
    return sum(1 for _ in frame_dir.glob(f"*.{image_ext}"))


def load_frame_mask(path: Path) -> np.ndarray:
    """Load a dataset mask as one 0/1 value per raw frame."""

    suffix = path.suffix.lower()
    if suffix == ".mat":
        return load_avenue_frame_mask(path)
    if suffix != ".npy":
        raise ValueError(f"Unsupported frame mask format: {path}")

    mask = np.load(path)
    if mask.ndim == 0:
        raise ValueError(f"Frame mask must have at least one dimension: {path}")
    if mask.ndim > 1:
        mask = mask.reshape(mask.shape[0], -1).max(axis=1)
    return (mask > 0).astype(np.uint8)


def load_avenue_frame_mask(path: Path) -> np.ndarray:
    """Load CUHK Avenue volLabel cell masks as frame-level 0/1 labels."""

    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise RuntimeError("scipy is required to read Avenue .mat ground-truth masks") from error

    mat = loadmat(path, squeeze_me=True)
    if "volLabel" not in mat:
        raise ValueError(f"Avenue label .mat must contain 'volLabel': {path}")

    masks = np.asarray(mat["volLabel"], dtype=object).reshape(-1)
    if masks.size == 0:
        raise ValueError(f"Avenue label .mat contains no masks: {path}")

    labels = []
    for mask in masks:
        mask_array = np.asarray(mask)
        if mask_array.ndim == 0:
            raise ValueError(f"Avenue per-frame mask must be an array: {path}")
        labels.append(int(np.any(mask_array > 0)))
    return np.asarray(labels, dtype=np.uint8)


def write_frame_index_map(
    writer: csv.DictWriter,
    record: VideoRecord,
    prepared: PreparedVideo,
) -> None:
    """Write raw-to-processed frame map rows."""

    for frame_map in prepared.frame_maps:
        writer.writerow(
            {
                "video_id": record.video_id,
                "split": record.split,
                "processed_frame_idx": frame_map.processed_frame_idx,
                "processed_frame_path": str(frame_map.processed_frame_path),
                "raw_frame_idx": blank_if_none(frame_map.raw_frame_idx),
                "raw_frame_path": blank_if_none(frame_map.raw_frame_path),
                "timestamp_sec": format_optional_float(frame_map.timestamp_sec),
            }
        )


def write_frame_labels(
    writer: csv.DictWriter,
    record: VideoRecord,
    prepared: PreparedVideo,
    config: PreprocessConfig,
) -> None:
    """Write frame-level labels for one processed sequence."""

    labels = labels_for_record(record, prepared, config)
    for frame_map, label in zip(prepared.frame_maps, labels.tolist(), strict=True):
        writer.writerow(
            {
                "video_id": record.video_id,
                "split": record.split,
                "class_name": record.class_name,
                "frame_idx": frame_map.processed_frame_idx,
                "frame_path": str(frame_map.processed_frame_path),
                "label": int(label),
            }
        )


def labels_for_record(
    record: VideoRecord,
    prepared: PreparedVideo,
    config: PreprocessConfig,
) -> np.ndarray:
    """Return processed-frame labels for one record."""

    if record.mask_path is None:
        return np.zeros(prepared.num_frames, dtype=np.uint8)

    raw_labels = load_frame_mask(record.mask_path)
    source_positions = [frame_map.source_position for frame_map in prepared.frame_maps]
    if all(position is not None for position in source_positions):
        positions = [int(position) for position in source_positions]
        if positions and max(positions) < len(raw_labels):
            return raw_labels[positions]

    if len(raw_labels) == prepared.num_frames:
        return raw_labels

    if config.label_mismatch_policy == "error":
        raise ValueError(
            f"Frame/mask length mismatch for {record.video_id}: "
            f"{prepared.num_frames} frames vs {len(raw_labels)} mask values"
        )
    raise ValueError(f"Unsupported label mismatch policy: {config.label_mismatch_policy}")


def build_sample_indices(config: PreprocessConfig) -> tuple[int, int]:
    """Build train/test context-future window index files."""

    labels_by_video = load_frame_labels(config.frame_labels_path)
    train_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    with config.manifest_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    for row in progress_bar(rows, desc="build window index", unit="video"):
        num_frames = int(row["num_frames"])
        if not has_any_window(num_frames, config.sampling):
            continue

        source_split = row["split"]
        is_train = source_split == TRAIN_SPLIT
        active_stride = config.sampling.stride_train if is_train else config.sampling.stride_test
        video_id = row["video_id"]
        frame_dir = Path(row["frame_dir"])
        video_labels = labels_by_video.get(video_id, {})

        for context, future in context_future_windows(
            num_frames,
            config.sampling,
            stride=active_stride,
            frame_index_start=config.frame_index_start,
        ):
            future_labels = [int(video_labels.get(frame_idx, 0)) for frame_idx in future]
            sample_id = f"{video_id}_{context[0]:06d}_{future[0]:06d}"
            record = {
                "sample_id": sample_id,
                "video_id": video_id,
                "split": "train" if is_train else "test",
                "source_split": source_split,
                "class_name": row["class_name"],
                "scene_id": row["class_name"],
                "is_anomaly": int(row["is_anomaly"]),
                "context_start_frame": context[0],
                "context_end_frame": context[-1],
                "future_start_frame": future[0],
                "future_end_frame": future[-1],
                "context_frames": context,
                "future_frames": future,
                "future_frame_labels": future_labels,
                "context_frame_paths": frame_paths(frame_dir, context, config.image_ext),
                "future_frame_paths": frame_paths(frame_dir, future, config.image_ext),
                "future_label": int(any(future_labels)),
            }
            if is_train:
                train_records.append(record)
            else:
                test_records.append(record)

    write_jsonl(config.train_samples_path, train_records)
    write_jsonl(config.test_samples_path, test_records)
    return len(train_records), len(test_records)


def load_frame_labels(path: Path) -> dict[str, dict[int, int]]:
    """Load frame-level labels keyed by video id and processed frame index."""

    labels: dict[str, dict[int, int]] = {}
    with path.open("r", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            labels.setdefault(row["video_id"], {})[int(row["frame_idx"])] = int(row["label"])
    return labels


def has_any_window(num_frames: int, sampling: SamplingConfig) -> bool:
    """Return whether a sequence is long enough for at least one window."""

    required = sampling.context_span + sampling.gap + sampling.future_frames
    return num_frames >= required


def context_future_windows(
    num_frames: int,
    sampling: SamplingConfig,
    *,
    stride: int,
    frame_index_start: int = 1,
) -> Iterable[tuple[list[int], list[int]]]:
    """Yield context/future frame indices according to sampling settings."""

    max_context_start = (
        frame_index_start
        + num_frames
        - sampling.context_span
        - sampling.gap
        - sampling.future_frames
    )
    if max_context_start < frame_index_start:
        return

    for context_start in range(frame_index_start, max_context_start + 1, stride):
        if sampling.context_sampling == "dense":
            context = list(range(context_start, context_start + sampling.context_frames))
        else:
            context = sparse_uniform_indices(
                context_start,
                sampling.context_span,
                sampling.context_frames,
            )
        future_start = context_start + sampling.context_span + sampling.gap
        future = list(range(future_start, future_start + sampling.future_frames))
        yield context, future


def sparse_uniform_indices(start: int, span: int, count: int) -> list[int]:
    """Return ``count`` monotonic indices inside a frame span."""

    if count <= 0:
        raise ValueError("count must be positive")
    if span < count:
        raise ValueError("span must be >= count")
    if count == 1:
        return [start + span - 1]
    offsets = [round(index * (span - 1) / (count - 1)) for index in range(count)]
    if len(set(offsets)) != len(offsets):
        raise ValueError("sparse frame sampling produced duplicate frame indices")
    return [start + int(offset) for offset in offsets]


def frame_paths(frame_dir: Path, indices: Iterable[int], image_ext: str) -> list[str]:
    """Return processed frame paths for frame indices."""

    return [str(frame_dir / f"{frame_idx:06d}.{image_ext}") for frame_idx in indices]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write JSONL records with stable ASCII encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records."""

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def blank_if_none(value: object | None) -> object:
    """Return an empty CSV value for None."""

    return "" if value is None else value


def format_optional_float(value: float | None) -> str:
    """Format optional floats for CSV output."""

    if value is None:
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse preprocessing CLI arguments."""

    parser = argparse.ArgumentParser(description="Preprocess raw video anomaly datasets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/local.yaml"),
        help="Path to preprocessing YAML config.",
    )
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""

    args = parse_args(argv)
    config = load_config(args.config)
    run_preprocess(
        config,
        limit_videos=args.limit_videos,
        limit_per_split=args.limit_per_split,
        overwrite=True if args.overwrite else None,
    )


if __name__ == "__main__":
    main()
