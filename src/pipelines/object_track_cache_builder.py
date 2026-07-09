"""Pseudo detector/tracker cache builders for Phase 0 track-aware VAD."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

from src.pipelines.feature_eng_pipeline import read_jsonl

MODE_SYNTHETIC = "synthetic"
MODE_MOTION_PROPOSAL = "motion_proposal"
MODES = {MODE_SYNTHETIC, MODE_MOTION_PROPOSAL}


@dataclass(frozen=True)
class FrameRef:
    """One processed frame reference."""

    video_id: str
    frame_idx: int
    path: Path


@dataclass(frozen=True)
class ComponentProposal:
    """One connected-component motion proposal in pixel coordinates."""

    bbox_xyxy: tuple[int, int, int, int]
    area: int
    score: float

    @property
    def center_xy(self) -> tuple[float, float]:
        """Return component center in pixel coordinates."""

        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


@dataclass
class ActiveTrack:
    """Mutable active track state for simple greedy association."""

    track_id: str
    bbox_xyxy: tuple[int, int, int, int]
    center_xy: tuple[float, float]
    age: int
    missing: int = 0


def build_pseudo_track_cache(
    *,
    output_track_cache: Path,
    mode: str,
    config_path: Path | None = None,
    feature_index_path: Path | None = None,
    frames_root: Path | None = None,
    min_area: int = 32,
    max_area: int | None = None,
    diff_percentile: float = 95.0,
    diff_std_k: float = 1.0,
    max_missing: int = 2,
    iou_threshold: float = 0.1,
    center_distance_threshold: float = 0.2,
) -> list[dict[str, Any]]:
    """Build a raw detector/tracker JSONL cache in Phase 1 format."""

    if mode not in MODES:
        raise ValueError("mode must be synthetic or motion_proposal")
    frame_refs = collect_frame_refs(
        config_path=config_path,
        feature_index_path=feature_index_path,
        frames_root=frames_root,
    )
    if not frame_refs:
        raise ValueError("No processed frames found for pseudo track cache generation")
    grouped = group_frame_refs_by_video(frame_refs)
    records: list[dict[str, Any]] = []
    for video_id, refs in grouped.items():
        if mode == MODE_SYNTHETIC:
            records.extend(build_synthetic_video_records(video_id, refs))
        else:
            records.extend(
                build_motion_proposal_video_records(
                    video_id,
                    refs,
                    min_area=min_area,
                    max_area=max_area,
                    diff_percentile=diff_percentile,
                    diff_std_k=diff_std_k,
                    max_missing=max_missing,
                    iou_threshold=iou_threshold,
                    center_distance_threshold=center_distance_threshold,
                )
            )
    write_track_cache(output_track_cache, records)
    return records


def collect_frame_refs(
    *,
    config_path: Path | None,
    feature_index_path: Path | None,
    frames_root: Path | None,
) -> list[FrameRef]:
    """Collect processed frame references from feature index or frames root."""

    if feature_index_path is not None:
        refs = frame_refs_from_feature_index(feature_index_path)
        if refs:
            return refs
    root = frames_root or frames_root_from_config(config_path)
    if root is None:
        raise ValueError(
            "Provide --feature-index, --frames-root, or --config with dataset.processed_root"
        )
    return frame_refs_from_frames_root(root)


def frame_refs_from_feature_index(path: Path) -> list[FrameRef]:
    """Infer frame references from context/future frame paths inside a feature index."""

    records = read_jsonl(path)
    refs_by_key: dict[tuple[str, int], FrameRef] = {}
    for record in records:
        video_id = str(record["video_id"])
        for frame_idx, frame_path in zip(
            record.get("context_frames", []),
            record.get("context_frame_paths", []),
            strict=False,
        ):
            refs_by_key[(video_id, int(frame_idx))] = FrameRef(
                video_id=video_id,
                frame_idx=int(frame_idx),
                path=Path(frame_path),
            )
        for frame_idx, frame_path in zip(
            record.get("future_frames", []),
            record.get("future_frame_paths", []),
            strict=False,
        ):
            refs_by_key[(video_id, int(frame_idx))] = FrameRef(
                video_id=video_id,
                frame_idx=int(frame_idx),
                path=Path(frame_path),
            )
    return sorted(refs_by_key.values(), key=lambda item: (item.video_id, item.frame_idx))


def frames_root_from_config(config_path: Path | None) -> Path | None:
    """Return processed test frames root from config when available."""

    if config_path is None:
        return None
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    processed_root = config.get("dataset", {}).get("processed_root")
    if not processed_root:
        return None
    return Path(processed_root) / "frames" / "test"


def frame_refs_from_frames_root(root: Path) -> list[FrameRef]:
    """Collect frame references from ``frames/test/<video_id>`` style directories."""

    if not root.is_dir():
        raise FileNotFoundError(f"Missing frames root: {root}")
    refs = []
    for video_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for frame_path in sorted(list_image_files(video_dir), key=frame_sort_key):
            frame_idx = parse_frame_idx(frame_path)
            if frame_idx is None:
                continue
            refs.append(FrameRef(video_id=video_dir.name, frame_idx=frame_idx, path=frame_path))
    return sorted(refs, key=lambda item: (item.video_id, item.frame_idx))


def frame_refs_from_manifest(manifest_path: Path) -> list[FrameRef]:
    """Collect frame references from processed frame_index_map.csv."""

    refs = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            frame_path = Path(row["processed_frame_path"])
            refs.append(
                FrameRef(
                    video_id=row["video_id"],
                    frame_idx=int(row["processed_frame_idx"]),
                    path=frame_path,
                )
            )
    return sorted(refs, key=lambda item: (item.video_id, item.frame_idx))


def list_image_files(root: Path) -> list[Path]:
    """Return image files under one video directory."""

    valid_suffixes = {".jpg", ".jpeg", ".png"}
    return [path for path in root.iterdir() if path.suffix.lower() in valid_suffixes]


def frame_sort_key(path: Path) -> tuple[int, int | str]:
    """Sort numeric frame stems before non-numeric stems."""

    frame_idx = parse_frame_idx(path)
    return (0, frame_idx) if frame_idx is not None else (1, path.stem)


def parse_frame_idx(path: Path) -> int | None:
    """Parse a processed frame index from a filename stem."""

    return int(path.stem) if path.stem.isdigit() else None


def group_frame_refs_by_video(frame_refs: Iterable[FrameRef]) -> dict[str, list[FrameRef]]:
    """Group frame references by video id."""

    grouped: dict[str, list[FrameRef]] = {}
    for ref in frame_refs:
        grouped.setdefault(ref.video_id, []).append(ref)
    return {key: sorted(value, key=lambda item: item.frame_idx) for key, value in grouped.items()}


def build_synthetic_video_records(video_id: str, refs: list[FrameRef]) -> list[dict[str, Any]]:
    """Build deterministic fake track records for smoke tests only."""

    records = []
    for offset, ref in enumerate(refs):
        image = load_gray_image(ref.path)
        height, width = image.shape
        objects = []
        if offset % 2 == 0:
            box_width = max(2, width // 5)
            box_height = max(2, height // 5)
            max_x = max(1, width - box_width)
            max_y = max(1, height - box_height)
            x1 = int(round((offset % 10) / 10 * max_x))
            y1 = int(round(((offset // 2) % 10) / 10 * max_y))
            x2 = min(width, x1 + box_width)
            y2 = min(height, y1 + box_height)
            vx = (box_width / max(width, 1)) * 0.1
            objects.append(
                make_object_record(
                    track_id="synthetic_1",
                    bbox_xyxy=normalize_bbox((x1, y1, x2, y2), width=width, height=height),
                    score=1.0,
                    class_id=-1,
                    class_name="synthetic_object",
                    velocity_xy_norm=(vx, 0.0),
                    age=offset + 1,
                    is_interpolated=False,
                )
            )
        records.append(
            make_frame_record(
                video_id=video_id,
                frame_idx=ref.frame_idx,
                image_size=(height, width),
                objects=objects,
                cache_source=MODE_SYNTHETIC,
            )
        )
    return records


def build_motion_proposal_video_records(
    video_id: str,
    refs: list[FrameRef],
    *,
    min_area: int,
    max_area: int | None,
    diff_percentile: float,
    diff_std_k: float,
    max_missing: int,
    iou_threshold: float,
    center_distance_threshold: float,
) -> list[dict[str, Any]]:
    """Build raw track records from frame-difference motion proposals."""

    if min_area < 0:
        raise ValueError("min_area must be non-negative")
    if max_area is not None and max_area < min_area:
        raise ValueError("max_area must be >= min_area")
    if not 0 <= diff_percentile <= 100:
        raise ValueError("diff_percentile must be in [0, 100]")
    if max_missing < 0:
        raise ValueError("max_missing must be non-negative")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    if center_distance_threshold < 0:
        raise ValueError("center_distance_threshold must be non-negative")

    active_tracks: list[ActiveTrack] = []
    next_track_id = 1
    records = []
    previous_gray: np.ndarray | None = None

    for ref in refs:
        current_gray = load_gray_image(ref.path)
        height, width = current_gray.shape
        if previous_gray is None:
            proposals: list[ComponentProposal] = []
        else:
            proposals = motion_component_proposals(
                previous_gray,
                current_gray,
                min_area=min_area,
                max_area=max_area,
                diff_percentile=diff_percentile,
                diff_std_k=diff_std_k,
            )
        assigned, active_tracks, next_track_id = assign_tracks(
            proposals,
            active_tracks,
            next_track_id=next_track_id,
            image_size=(height, width),
            max_missing=max_missing,
            iou_threshold=iou_threshold,
            center_distance_threshold=center_distance_threshold,
        )
        records.append(
            make_frame_record(
                video_id=video_id,
                frame_idx=ref.frame_idx,
                image_size=(height, width),
                objects=[
                    make_object_record(
                        track_id=track.track_id,
                        bbox_xyxy=normalize_bbox(proposal.bbox_xyxy, width=width, height=height),
                        score=proposal.score,
                        class_id=-1,
                        class_name="unknown_moving_object",
                        velocity_xy_norm=velocity_xy_norm,
                        age=track.age,
                        is_interpolated=False,
                    )
                    for proposal, track, velocity_xy_norm in assigned
                ],
                cache_source=MODE_MOTION_PROPOSAL,
            )
        )
        previous_gray = current_gray
    return records


def load_gray_image(path: Path) -> np.ndarray:
    """Load one processed frame as grayscale uint8."""

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read frame image: {path}")
    return image


def motion_component_proposals(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    *,
    min_area: int,
    max_area: int | None,
    diff_percentile: float,
    diff_std_k: float,
) -> list[ComponentProposal]:
    """Return connected-component proposals from adjacent grayscale frame difference."""

    if previous_gray.shape != current_gray.shape:
        raise ValueError("Adjacent frames must have the same shape for motion proposals")
    diff = cv2.absdiff(previous_gray, current_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    threshold_value = max(
        float(np.percentile(diff, diff_percentile)), float(diff.mean() + diff_std_k * diff.std())
    )
    if threshold_value <= 0:
        return []
    _, mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    height, width = current_gray.shape
    proposals = []
    for label in range(1, num_labels):
        x, y, box_w, box_h, area = stats[label].tolist()
        if not component_is_valid(
            x=x,
            y=y,
            width=box_w,
            height=box_h,
            area=area,
            image_width=width,
            image_height=height,
            min_area=min_area,
            max_area=max_area,
        ):
            continue
        component_mask = labels[y : y + box_h, x : x + box_w] == label
        score = float(diff[y : y + box_h, x : x + box_w][component_mask].mean() / 255.0)
        proposals.append(
            ComponentProposal(
                bbox_xyxy=(x, y, x + box_w, y + box_h),
                area=int(area),
                score=max(0.0, min(1.0, score)),
            )
        )
    return sorted(proposals, key=lambda item: item.area, reverse=True)


def component_is_valid(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    area: int,
    image_width: int,
    image_height: int,
    min_area: int,
    max_area: int | None,
) -> bool:
    """Return whether a connected component passes simple geometric filters."""

    if area < min_area:
        return False
    if max_area is not None and area > max_area:
        return False
    if width <= 1 or height <= 1:
        return False
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        return False
    aspect = width / max(height, 1)
    if aspect < 0.1 or aspect > 10.0:
        return False
    return True


def assign_tracks(
    proposals: list[ComponentProposal],
    active_tracks: list[ActiveTrack],
    *,
    next_track_id: int,
    image_size: tuple[int, int],
    max_missing: int,
    iou_threshold: float,
    center_distance_threshold: float,
) -> tuple[
    list[tuple[ComponentProposal, ActiveTrack, tuple[float, float]]], list[ActiveTrack], int
]:
    """Greedily associate proposals to active tracks by IoU and center distance."""

    height, width = image_size
    assignments: list[tuple[ComponentProposal, ActiveTrack, tuple[float, float]]] = []
    unmatched_tracks = active_tracks[:]
    updated_tracks: list[ActiveTrack] = []

    for proposal in proposals:
        match_index = best_track_match(
            proposal,
            unmatched_tracks,
            image_size=(height, width),
            iou_threshold=iou_threshold,
            center_distance_threshold=center_distance_threshold,
        )
        if match_index is None:
            track = ActiveTrack(
                track_id=str(next_track_id),
                bbox_xyxy=proposal.bbox_xyxy,
                center_xy=proposal.center_xy,
                age=1,
                missing=0,
            )
            velocity_xy_norm = (0.0, 0.0)
            next_track_id += 1
        else:
            previous = unmatched_tracks.pop(match_index)
            velocity_xy_norm = velocity_from_centers(
                previous.center_xy,
                proposal.center_xy,
                width=width,
                height=height,
            )
            track = ActiveTrack(
                track_id=previous.track_id,
                bbox_xyxy=proposal.bbox_xyxy,
                center_xy=proposal.center_xy,
                age=previous.age + 1,
                missing=0,
            )
        assignments.append((proposal, track, velocity_xy_norm))
        updated_tracks.append(track)

    for track in unmatched_tracks:
        if track.missing + 1 <= max_missing:
            track.missing += 1
            updated_tracks.append(track)
    return assignments, updated_tracks, next_track_id


def best_track_match(
    proposal: ComponentProposal,
    tracks: list[ActiveTrack],
    *,
    image_size: tuple[int, int],
    iou_threshold: float,
    center_distance_threshold: float,
) -> int | None:
    """Return the best active-track index for one proposal."""

    height, width = image_size
    diagonal = math.sqrt(width * width + height * height)
    best_index = None
    best_score = -1.0
    for index, track in enumerate(tracks):
        current_iou = bbox_iou(proposal.bbox_xyxy, track.bbox_xyxy)
        center_dist = normalized_center_distance(proposal.center_xy, track.center_xy, diagonal)
        if current_iou < iou_threshold and center_dist > center_distance_threshold:
            continue
        score = current_iou - center_dist
        if score > best_score:
            best_index = index
            best_score = score
    return best_index


def bbox_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    """Return IoU for two pixel-space xyxy boxes."""

    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def normalized_center_distance(
    first: tuple[float, float],
    second: tuple[float, float],
    diagonal: float,
) -> float:
    """Return center distance normalized by image diagonal."""

    if diagonal <= 0:
        return 0.0
    return math.dist(first, second) / diagonal


def velocity_from_centers(
    previous_center_xy: tuple[float, float],
    current_center_xy: tuple[float, float],
    *,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Return center delta normalized by image size for the current assignment."""

    px, py = previous_center_xy
    cx, cy = current_center_xy
    return ((cx - px) / max(width, 1), (cy - py) / max(height, 1))


def normalize_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
) -> list[float]:
    """Normalize and validate a pixel-space xyxy bbox."""

    x1, y1, x2, y2 = bbox_xyxy
    x1_norm = clamp01(x1 / max(width, 1))
    y1_norm = clamp01(y1 / max(height, 1))
    x2_norm = clamp01(x2 / max(width, 1))
    y2_norm = clamp01(y2 / max(height, 1))
    if not (x1_norm < x2_norm and y1_norm < y2_norm):
        raise ValueError(f"Invalid normalized bbox from {bbox_xyxy}")
    return [x1_norm, y1_norm, x2_norm, y2_norm]


def clamp01(value: float) -> float:
    """Clamp a float to [0, 1]."""

    return max(0.0, min(1.0, float(value)))


def make_object_record(
    *,
    track_id: str,
    bbox_xyxy: list[float],
    score: float,
    class_id: int,
    class_name: str,
    velocity_xy_norm: tuple[float, float],
    age: int,
    is_interpolated: bool,
) -> dict[str, Any]:
    """Return one raw cache object record."""

    return {
        "track_id": str(track_id),
        "bbox_xyxy_norm": [float(value) for value in bbox_xyxy],
        "score": float(max(0.0, min(1.0, score))),
        "class_id": int(class_id),
        "class_name": class_name,
        "velocity_xy_norm": [float(velocity_xy_norm[0]), float(velocity_xy_norm[1])],
        "age": int(age),
        "is_interpolated": bool(is_interpolated),
    }


def make_frame_record(
    *,
    video_id: str,
    frame_idx: int,
    image_size: tuple[int, int],
    objects: list[dict[str, Any]],
    cache_source: str,
) -> dict[str, Any]:
    """Return one raw track cache frame record."""

    height, width = image_size
    return {
        "video_id": str(video_id),
        "frame_idx": int(frame_idx),
        "image_size": [int(height), int(width)],
        "objects": objects,
        "cache_source": cache_source,
    }


def write_track_cache(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write raw detector/tracker cache JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse pseudo track cache CLI arguments."""

    parser = argparse.ArgumentParser(description="Build pseudo detector/tracker raw cache JSONL.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--feature-index", type=Path, default=None)
    parser.add_argument("--frames-root", type=Path, default=None)
    parser.add_argument("--output-track-cache", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--min-area", type=int, default=32)
    parser.add_argument("--max-area", type=int, default=None)
    parser.add_argument("--diff-percentile", type=float, default=95.0)
    parser.add_argument("--diff-std-k", type=float, default=1.0)
    parser.add_argument("--max-missing", type=int, default=2)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--center-distance-threshold", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""

    args = parse_args(argv)
    records = build_pseudo_track_cache(
        output_track_cache=args.output_track_cache,
        mode=args.mode,
        config_path=args.config,
        feature_index_path=args.feature_index,
        frames_root=args.frames_root,
        min_area=args.min_area,
        max_area=args.max_area,
        diff_percentile=args.diff_percentile,
        diff_std_k=args.diff_std_k,
        max_missing=args.max_missing,
        iou_threshold=args.iou_threshold,
        center_distance_threshold=args.center_distance_threshold,
    )
    print(f"wrote {len(records)} pseudo track cache records -> {args.output_track_cache}")


if __name__ == "__main__":
    main(sys.argv[1:])
