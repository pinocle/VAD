"""Small MA-PDM-style frame preprocessing for the RGB-patch VAD pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image

from src.utils import configure_progress_from_config, progress_bar

TRAINING_PHASE = "training"
TESTING_PHASE = "testing"
MA_PDM_PHASES = (TRAINING_PHASE, TESTING_PHASE)
MA_PDM_DATASETS = {"shanghai", "avenue", "ped2", "ucf", "xd"}
DATASET_ALIASES = {
    "shanghaitech": "shanghai",
    "shanghaitech_campus": "shanghai",
    "cuhk_avenue": "avenue",
    "ucfcrime": "ucf",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class SamplingConfig:
    """Dense MA-PDM-style context/future window settings."""

    context_frames: int
    future_frames: int
    gap: int
    stride_train: int
    stride_test: int


@dataclass(frozen=True)
class PreprocessConfig:
    """Resolved preprocessing settings."""

    dataset_name: str
    raw_root: Path
    processed_root: Path
    image_ext: str
    jpeg_quality: int
    frame_index_start: int
    overwrite: bool
    train_split_path: Path | None
    test_split_path: Path | None
    test_labels_path: Path | None
    sampling: SamplingConfig

    @property
    def raw_dataset_root(self) -> Path:
        return resolve_dataset_root(self.raw_root, self.dataset_name)

    @property
    def processed_dataset_root(self) -> Path:
        return resolve_dataset_root(self.processed_root, self.dataset_name)

    @property
    def metadata_root(self) -> Path:
        return self.processed_dataset_root / "metadata"

    @property
    def manifest_path(self) -> Path:
        return self.metadata_root / "manifest.jsonl"

    @property
    def frame_labels_path(self) -> Path:
        return self.metadata_root / "frame_labels.jsonl"

    @property
    def train_samples_path(self) -> Path:
        return self.metadata_root / "samples_train.jsonl"

    @property
    def test_samples_path(self) -> Path:
        return self.metadata_root / "samples_test.jsonl"

    def raw_frames_root(self, phase: str) -> Path:
        return self.raw_dataset_root / phase / "frames"

    def frames_root(self, phase: str) -> Path:
        return self.processed_dataset_root / phase / "frames"


@dataclass(frozen=True)
class VideoRecord:
    """One MA-PDM video directory selected for preprocessing."""

    video_id: str
    phase: str
    frame_dir: Path
    labels: np.ndarray | None


@dataclass(frozen=True)
class ProcessedVideo:
    """Processed frame paths and aligned labels for one video."""

    video_id: str
    phase: str
    frames: tuple[Path, ...]
    labels: np.ndarray


def load_config(path: Path) -> PreprocessConfig:
    """Load config and keep only options used by the simple frame preprocessor."""

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    configure_progress_from_config(raw)

    dataset = dict(raw.get("dataset", {}))
    preprocess = dict(raw.get("preprocess", {}))
    sampling = load_sampling_config(dict(raw.get("sampling", {})))
    dataset_name = normalize_dataset_name(dataset.get("name"))
    image_ext = str(preprocess.get("image_ext", "jpg")).lower().lstrip(".")
    if f".{image_ext}" not in IMAGE_SUFFIXES:
        raise ValueError("preprocess.image_ext must be jpg, jpeg, png, or bmp")
    jpeg_quality = int(preprocess.get("jpeg_quality", 95))
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("preprocess.jpeg_quality must be in [1, 100]")

    return PreprocessConfig(
        dataset_name=dataset_name,
        raw_root=Path(dataset.get("raw_root", "data/01_raw")),
        processed_root=Path(dataset.get("processed_root", "data/02_processed")),
        image_ext=image_ext,
        jpeg_quality=jpeg_quality,
        frame_index_start=resolve_frame_index_start(
            dataset_name,
            preprocess.get("frame_index_start", "auto"),
        ),
        overwrite=bool(preprocess.get("overwrite", False)),
        train_split_path=parse_optional_path(dataset.get("train_split_path")),
        test_split_path=parse_optional_path(dataset.get("test_split_path")),
        test_labels_path=parse_optional_path(dataset.get("test_labels_path")),
        sampling=sampling,
    )


def normalize_dataset_name(value: Any) -> str:
    """Normalize the dataset name to MA-PDM's canonical names."""

    name = str(value or "").strip().lower()
    name = DATASET_ALIASES.get(name, name)
    if name not in MA_PDM_DATASETS:
        supported = ", ".join(sorted(MA_PDM_DATASETS))
        raise ValueError(f"dataset.name must be one of: {supported}")
    return name


def parse_optional_path(value: Any) -> Path | None:
    """Return a path only when the config value is present and non-empty."""

    if value is None:
        return None
    parsed = str(value).strip()
    return Path(parsed) if parsed else None


def resolve_frame_index_start(dataset_name: str, value: Any) -> int:
    """Match MA-PDM indexing: Shanghai frames start at 1, others at 0."""

    if isinstance(value, str) and value.strip().lower() == "auto":
        return 1 if dataset_name == "shanghai" else 0
    start = int(value)
    if start < 0:
        raise ValueError("preprocess.frame_index_start must be non-negative or auto")
    return start


def resolve_dataset_root(root: Path, dataset_name: str) -> Path:
    """Accept either MA-PDM's video_folder root or a direct dataset root."""

    if root.name == dataset_name:
        return root
    if (root / TRAINING_PHASE / "frames").exists() or (root / TESTING_PHASE / "frames").exists():
        return root
    return root / dataset_name


def load_sampling_config(raw: dict[str, Any]) -> SamplingConfig:
    """Load dense windows equivalent to MA-PDM's time_step + num_pred scan."""

    context_frames = int(raw.get("context_frames", raw.get("time_step", 32)))
    future_frames = int(raw.get("future_frames", raw.get("num_pred", 8)))
    gap = int(raw.get("gap", 0))
    stride_train = int(raw.get("stride_train", 1))
    stride_test = int(raw.get("stride_test", 1))
    context_sampling = str(raw.get("context_sampling", "dense"))
    context_span = int(raw.get("context_span", context_frames))

    if min(context_frames, future_frames, stride_train, stride_test) <= 0:
        raise ValueError("sampling frame counts and strides must be positive")
    if gap < 0:
        raise ValueError("sampling.gap must be non-negative")
    if context_sampling != "dense" or context_span != context_frames:
        raise ValueError("MA-PDM-style preprocessing supports dense contiguous context only")
    return SamplingConfig(
        context_frames=context_frames,
        future_frames=future_frames,
        gap=gap,
        stride_train=stride_train,
        stride_test=stride_test,
    )


def run_preprocess(
    config: PreprocessConfig,
    *,
    phases: set[str] | None = None,
    limit_videos: int | None = None,
    overwrite: bool | None = None,
) -> tuple[int, int]:
    """Copy MA-PDM frame trees to ``02_processed`` and write sample windows."""

    records = discover_video_records(config, phases=phases)
    if limit_videos is not None:
        if limit_videos <= 0:
            raise ValueError("limit_videos must be positive")
        records = records[:limit_videos]
    if not records:
        raise ValueError("No videos selected for preprocessing")

    active_overwrite = config.overwrite if overwrite is None else overwrite
    processed = [
        process_video(record, config, overwrite=active_overwrite)
        for record in progress_bar(records, desc="preprocess frames", unit="video")
    ]
    write_metadata(config, processed)
    return build_sample_indices(config, processed)


def discover_video_records(
    config: PreprocessConfig,
    *,
    phases: set[str] | None,
) -> list[VideoRecord]:
    """Discover ``<root>/<dataset>/<phase>/frames/<video>`` directories."""

    selected = set(MA_PDM_PHASES if phases is None else phases)
    unknown = selected - set(MA_PDM_PHASES)
    if unknown:
        raise ValueError(f"Unsupported phases: {sorted(unknown)}")

    records = []
    for phase in MA_PDM_PHASES:
        if phase not in selected:
            continue
        root = config.raw_frames_root(phase)
        if not root.is_dir():
            raise FileNotFoundError(f"Missing MA-PDM frames directory: {root}")
        allowed = load_split_filter(
            config.train_split_path if phase == TRAINING_PHASE else config.test_split_path
        )
        for frame_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if allowed is not None and frame_dir.name not in allowed:
                continue
            records.append(
                VideoRecord(
                    video_id=frame_dir.name,
                    phase=phase,
                    frame_dir=frame_dir,
                    labels=load_labels(config, frame_dir.name, phase),
                )
            )
    return records


def load_split_filter(path: Path | None) -> set[str] | None:
    """Read the same JSON split shape used by MA-PDM for UCF/XD."""

    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        return {str(key) for key in value}
    if isinstance(value, list):
        return {str(item) for item in value}
    raise ValueError("split JSON must be an object or array")


def load_labels(config: PreprocessConfig, video_id: str, phase: str) -> np.ndarray | None:
    """Load optional test labels; training videos remain normal by convention."""

    if phase == TRAINING_PHASE:
        return None
    if config.test_labels_path is not None:
        return load_label_source(config.test_labels_path, video_id)
    shanghai_mask = config.raw_dataset_root / TESTING_PHASE / "test_frame_mask" / f"{video_id}.npy"
    if config.dataset_name == "shanghai" and shanghai_mask.is_file():
        return normalize_labels(np.load(shanghai_mask))
    return None


def load_label_source(path: Path, video_id: str) -> np.ndarray:
    """Load labels from ``.npy``, a per-video directory, or a JSON mapping/list."""

    if path.is_dir():
        for suffix in (".npy", ".json"):
            candidate = path / f"{video_id}{suffix}"
            if candidate.is_file():
                return load_label_source(candidate, video_id)
        raise FileNotFoundError(f"Missing labels for {video_id} in {path}")
    if path.suffix.lower() == ".npy":
        return normalize_labels(np.load(path))
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            if video_id not in value:
                raise ValueError(f"Label JSON does not contain video {video_id}")
            value = value[video_id]
        return normalize_labels(np.asarray(value))
    raise ValueError("test_labels_path must be a .npy, .json, or directory")


def normalize_labels(labels: np.ndarray) -> np.ndarray:
    """Reduce masks or vectors to one binary label per frame."""

    values = np.asarray(labels)
    if values.ndim == 0:
        raise ValueError("Frame labels must have at least one dimension")
    if values.ndim > 1:
        values = values.reshape(values.shape[0], -1).max(axis=1)
    return (values > 0).astype(np.uint8)


def process_video(
    record: VideoRecord,
    config: PreprocessConfig,
    *,
    overwrite: bool,
) -> ProcessedVideo:
    """Copy one frame directory into the processed MA-PDM layout."""

    target_dir = config.frames_root(record.phase) / record.video_id
    existing = output_frame_paths(target_dir, config.image_ext)
    if existing and not overwrite:
        labels = align_labels(record.labels, len(existing), record.video_id)
        return ProcessedVideo(record.video_id, record.phase, tuple(existing), labels)

    source_frames = source_frame_paths(record.frame_dir)
    if record.frame_dir != target_dir:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        frames = copy_frame_directory(source_frames, target_dir, config)
    else:
        frames = source_frames

    labels = align_labels(record.labels, len(frames), record.video_id)
    return ProcessedVideo(record.video_id, record.phase, tuple(frames), labels)


def source_frame_paths(directory: Path) -> list[Path]:
    """Return sorted image files from a MA-PDM video frame directory."""

    if not directory.is_dir():
        raise FileNotFoundError(f"Missing frame directory: {directory}")
    frames = [path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
    if not frames:
        raise ValueError(f"No image frames found in {directory}")
    return sorted(frames, key=frame_sort_key)


def output_frame_paths(directory: Path, image_ext: str) -> list[Path]:
    """Return sorted processed frames with the configured extension."""

    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.glob(f"*.{image_ext}") if path.is_file()),
        key=frame_sort_key,
    )


def frame_sort_key(path: Path) -> tuple[int, int | str]:
    """Sort numeric frame names before falling back to lexical order."""

    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name)


def copy_frame_directory(
    source_frames: list[Path],
    target_dir: Path,
    config: PreprocessConfig,
) -> list[Path]:
    """Copy frames to MA-PDM's numeric filename convention."""

    output_paths = []
    for position, source in enumerate(source_frames):
        frame_idx = config.frame_index_start + position
        target = target_dir / f"{frame_idx:06d}.{config.image_ext}"
        copy_or_convert_frame(source, target, config)
        output_paths.append(target)
    return output_paths


def copy_or_convert_frame(source: Path, target: Path, config: PreprocessConfig) -> None:
    """Copy images when possible and convert only when the extension changes."""

    if source == target:
        return
    if source.suffix.lower() == f".{config.image_ext}":
        shutil.copy2(source, target)
        return
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        save_kwargs: dict[str, Any] = {}
        if config.image_ext in {"jpg", "jpeg"}:
            save_kwargs["quality"] = config.jpeg_quality
        rgb.save(target, **save_kwargs)


def align_labels(labels: np.ndarray | None, frame_count: int, video_id: str) -> np.ndarray:
    """Return zero labels or validate one label per processed frame."""

    if labels is None:
        return np.zeros(frame_count, dtype=np.uint8)
    labels = normalize_labels(labels)
    if len(labels) != frame_count:
        raise ValueError(
            f"Label count for {video_id} does not match frames: "
            f"{len(labels)} labels vs {frame_count} frames"
        )
    return labels


def write_metadata(config: PreprocessConfig, videos: list[ProcessedVideo]) -> None:
    """Write the small metadata set needed by training and inference."""

    manifest = []
    frame_labels = []
    for video in videos:
        manifest.append(
            {
                "video_id": video.video_id,
                "phase": video.phase,
                "frame_dir": str(config.frames_root(video.phase) / video.video_id),
                "num_frames": len(video.frames),
                "has_anomaly": int(video.labels.any()),
            }
        )
        for position, label in enumerate(video.labels.tolist()):
            frame_labels.append(
                {
                    "video_id": video.video_id,
                    "phase": video.phase,
                    "frame_idx": config.frame_index_start + position,
                    "label": int(label),
                }
            )

    write_jsonl(config.manifest_path, manifest)
    write_jsonl(config.frame_labels_path, frame_labels)
    layout = {
        "dataset": config.dataset_name,
        "layout": "ma_pdm",
        "frames": "<root>/<dataset>/{training,testing}/frames/<video_id>/<frame>.jpg",
        "frame_index_start": config.frame_index_start,
        "image_ext": config.image_ext,
        "sampling": asdict(config.sampling),
    }
    config.metadata_root.mkdir(parents=True, exist_ok=True)
    (config.metadata_root / "layout.json").write_text(
        json.dumps(layout, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_sample_indices(
    config: PreprocessConfig,
    videos: list[ProcessedVideo],
) -> tuple[int, int]:
    """Write dense sample windows using MA-PDM's sequential scan pattern."""

    train_records = []
    test_records = []
    for video in progress_bar(videos, desc="build samples", unit="video"):
        stride = (
            config.sampling.stride_train
            if video.phase == TRAINING_PHASE
            else config.sampling.stride_test
        )
        for context_positions, future_positions in context_future_windows(
            len(video.frames),
            config.sampling,
            stride=stride,
        ):
            context_paths = [video.frames[position] for position in context_positions]
            future_paths = [video.frames[position] for position in future_positions]
            future_labels = [int(video.labels[position]) for position in future_positions]
            context_ids = [config.frame_index_start + position for position in context_positions]
            future_ids = [config.frame_index_start + position for position in future_positions]
            record = {
                "sample_id": f"{video.video_id}_{context_ids[0]:06d}_{future_ids[0]:06d}",
                "video_id": video.video_id,
                "split": "train" if video.phase == TRAINING_PHASE else "test",
                "phase": video.phase,
                "context_frames": context_ids,
                "future_frames": future_ids,
                "future_frame_labels": future_labels,
                "future_label": int(any(future_labels)),
                "context_frame_paths": [str(path) for path in context_paths],
                "future_frame_paths": [str(path) for path in future_paths],
            }
            if video.phase == TRAINING_PHASE:
                train_records.append(record)
            else:
                test_records.append(record)

    write_jsonl(config.train_samples_path, train_records)
    write_jsonl(config.test_samples_path, test_records)
    return len(train_records), len(test_records)


def context_future_windows(
    frame_count: int,
    sampling: SamplingConfig,
    *,
    stride: int,
) -> Iterable[tuple[list[int], list[int]]]:
    """Yield dense context positions followed by contiguous future positions."""

    window = sampling.context_frames + sampling.gap + sampling.future_frames
    last_start = frame_count - window
    if last_start < 0:
        return
    for start in range(0, last_start + 1, stride):
        context = list(range(start, start + sampling.context_frames))
        future_start = start + sampling.context_frames + sampling.gap
        future = list(range(future_start, future_start + sampling.future_frames))
        yield context, future


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write UTF-8 JSONL records atomically at the file level."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file."""

    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse preprocessing CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/local.yaml"))
    parser.add_argument("--phase", choices=[*MA_PDM_PHASES, "all"], default="all")
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run preprocessing from the command line."""

    args = parse_args(argv)
    config = load_config(args.config)
    phases = None if args.phase == "all" else {args.phase}
    train_count, test_count = run_preprocess(
        config,
        phases=phases,
        limit_videos=args.limit_videos,
        overwrite=args.overwrite,
    )
    print(
        f"wrote MA-PDM frames and RGB-patch samples -> {config.processed_dataset_root} "
        f"(train={train_count}, test={test_count})"
    )


if __name__ == "__main__":
    main()
