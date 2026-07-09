"""Detector/tracker raw cache builders for Phase 1.5 track-aware VAD."""

from __future__ import annotations

import argparse
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
from src.pipelines.object_track_cache_builder import (
    ActiveTrack,
    FrameRef,
    bbox_iou,
    collect_frame_refs,
    component_is_valid,
    group_frame_refs_by_video,
    make_frame_record,
    make_object_record,
    normalize_bbox,
    normalized_center_distance,
    velocity_from_centers,
    write_track_cache,
)

MODE_DETECTOR_ONLY = "detector_only"
MODE_DETECTOR_TRACKER = "detector_tracker"
MODE_ORACLE_MASK = "oracle_mask"
MODES = {MODE_DETECTOR_ONLY, MODE_DETECTOR_TRACKER, MODE_ORACLE_MASK}

DETECTOR_FIXTURE_JSONL = "fixture_jsonl"
DETECTOR_ULTRALYTICS = "ultralytics"
DETECTORS = {DETECTOR_FIXTURE_JSONL, DETECTOR_ULTRALYTICS}

TRACKER_NONE = "none"
TRACKER_SIMPLE_IOU = "simple_iou"
TRACKER_BYTETRACK = "bytetrack"
TRACKER_OCSORT = "ocsort"
TRACKERS = {TRACKER_NONE, TRACKER_SIMPLE_IOU, TRACKER_BYTETRACK, TRACKER_OCSORT}

CACHE_SOURCE_DETECTOR_ONLY = "detector_only"
CACHE_SOURCE_DETECTOR_TRACKER = "detector_tracker"
CACHE_SOURCE_ORACLE_MASK = "oracle_mask"

COARSE_CLASS_IDS = {
    "person": 0,
    "vehicle": 1,
    "two_wheeler": 2,
    "carried_object": 3,
    "other_object": 4,
    "unknown": 5,
}

DEFAULT_CLASS_REMAP = {
    "person": "person",
    "pedestrian": "person",
    "rider": "person",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "train": "vehicle",
    "van": "vehicle",
    "vehicle": "vehicle",
    "bicycle": "two_wheeler",
    "bike": "two_wheeler",
    "motorcycle": "two_wheeler",
    "motorbike": "two_wheeler",
    "scooter": "two_wheeler",
    "backpack": "carried_object",
    "handbag": "carried_object",
    "suitcase": "carried_object",
    "bag": "carried_object",
    "umbrella": "carried_object",
    "unknown": "unknown",
}


@dataclass(frozen=True)
class Detection:
    """One pixel-space detector output after backend parsing."""

    bbox_xyxy: tuple[int, int, int, int]
    score: float
    class_id: int
    class_name: str

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


@dataclass(frozen=True)
class PreparedDetection:
    """One filtered detector output ready for raw-cache serialization."""

    bbox_xyxy: tuple[int, int, int, int]
    score: float
    raw_class_id: int
    raw_class_name: str
    coarse_class_name: str

    @property
    def coarse_class_id(self) -> int:
        return COARSE_CLASS_IDS[self.coarse_class_name]

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


class DetectorBackend:
    """Small detector backend protocol."""

    name: str

    def detect(self, ref: FrameRef, image_bgr: np.ndarray) -> list[Detection]:
        raise NotImplementedError


class FixtureJsonlDetector(DetectorBackend):
    """Detector backend backed by JSONL fixtures or external detector dumps."""

    name = DETECTOR_FIXTURE_JSONL

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Missing fixture detector JSONL: {path}")
        self.detections_by_frame = load_fixture_detections(path)

    def detect(self, ref: FrameRef, image_bgr: np.ndarray) -> list[Detection]:
        height, width = image_bgr.shape[:2]
        detections = []
        for raw in self.detections_by_frame.get((ref.video_id, ref.frame_idx), []):
            detections.append(parse_fixture_detection(raw, width=width, height=height))
        return detections


class UltralyticsDetector(DetectorBackend):
    """Optional Ultralytics YOLO detector backend."""

    name = DETECTOR_ULTRALYTICS

    def __init__(
        self,
        *,
        model_path: str | None,
        confidence_threshold: float,
        nms_threshold: float,
        device: str | None,
    ) -> None:
        if not model_path:
            raise RuntimeError(
                "detector_name=ultralytics requires --detector-model, for example "
                "yolov8n.pt or a local YOLO checkpoint path."
            )
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "detector_name=ultralytics requires optional dependency 'ultralytics'. "
                "Install it with: pip install ultralytics"
            ) from error
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device

    def detect(self, ref: FrameRef, image_bgr: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            source=str(ref.path),
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            device=self.device,
            verbose=False,
        )[0]
        names = getattr(result, "names", {}) or {}
        detections = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy()
        for bbox, score, class_id in zip(xyxy, scores, classes, strict=False):
            class_id_int = int(class_id)
            class_name = str(names.get(class_id_int, class_id_int))
            detections.append(
                Detection(
                    bbox_xyxy=tuple(int(round(float(value))) for value in bbox),  # type: ignore[arg-type]
                    score=float(score),
                    class_id=class_id_int,
                    class_name=class_name,
                )
            )
        return detections


def build_detector_track_cache(
    *,
    output_track_cache: Path,
    mode: str,
    config_path: Path | None = None,
    feature_index_path: Path | None = None,
    frames_root: Path | None = None,
    detector_name: str = DETECTOR_ULTRALYTICS,
    detector_model: str | None = None,
    fixture_detections_path: Path | None = None,
    tracker_name: str = TRACKER_SIMPLE_IOU,
    mask_root: Path | None = None,
    class_map_path: Path | None = None,
    confidence_threshold: float = 0.25,
    nms_threshold: float = 0.7,
    min_box_area: int = 16,
    max_objects_per_frame: int = 100,
    max_missing: int = 2,
    iou_threshold: float = 0.1,
    center_distance_threshold: float = 0.2,
    processed_frame_index_base: int | None = None,
    debug_output_dir: Path | None = None,
    debug_max_frames: int = 0,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Build detector/tracker raw cache JSONL records."""

    validate_build_args(
        mode=mode,
        detector_name=detector_name,
        tracker_name=tracker_name,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        min_box_area=min_box_area,
        max_objects_per_frame=max_objects_per_frame,
        max_missing=max_missing,
        iou_threshold=iou_threshold,
        center_distance_threshold=center_distance_threshold,
    )
    frame_refs = collect_frame_refs(
        config_path=config_path,
        feature_index_path=feature_index_path,
        frames_root=frames_root,
    )
    if not frame_refs:
        raise ValueError("No processed frames found for detector/tracker cache generation")
    index_base = (
        int(processed_frame_index_base)
        if processed_frame_index_base is not None
        else infer_processed_frame_index_base(frame_refs)
    )
    class_map = load_class_remap(class_map_path)

    if mode == MODE_ORACLE_MASK:
        records = build_oracle_mask_records(
            frame_refs,
            mask_root=mask_root,
            class_map=class_map,
            min_box_area=min_box_area,
            max_objects_per_frame=max_objects_per_frame,
            processed_frame_index_base=index_base,
        )
    else:
        detector = create_detector(
            detector_name=detector_name,
            detector_model=detector_model,
            fixture_detections_path=fixture_detections_path,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            device=device,
        )
        if mode == MODE_DETECTOR_ONLY:
            records = build_detector_only_records(
                frame_refs,
                detector=detector,
                class_map=class_map,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                min_box_area=min_box_area,
                max_objects_per_frame=max_objects_per_frame,
                processed_frame_index_base=index_base,
            )
        else:
            records = build_detector_tracker_records(
                frame_refs,
                detector=detector,
                tracker_name=tracker_name,
                class_map=class_map,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                min_box_area=min_box_area,
                max_objects_per_frame=max_objects_per_frame,
                max_missing=max_missing,
                iou_threshold=iou_threshold,
                center_distance_threshold=center_distance_threshold,
                processed_frame_index_base=index_base,
            )

    write_track_cache(output_track_cache, records)
    if debug_output_dir is not None and debug_max_frames > 0:
        write_debug_visualizations(
            records,
            frame_refs=frame_refs,
            output_dir=debug_output_dir,
            max_frames=debug_max_frames,
        )
    return records


def validate_build_args(
    *,
    mode: str,
    detector_name: str,
    tracker_name: str,
    confidence_threshold: float,
    nms_threshold: float,
    min_box_area: int,
    max_objects_per_frame: int,
    max_missing: int,
    iou_threshold: float,
    center_distance_threshold: float,
) -> None:
    """Validate detector/tracker cache generation arguments."""

    if mode not in MODES:
        raise ValueError("mode must be detector_only, detector_tracker, or oracle_mask")
    if detector_name not in DETECTORS:
        raise ValueError("detector_name must be fixture_jsonl or ultralytics")
    if tracker_name not in TRACKERS:
        raise ValueError("tracker_name must be none, simple_iou, bytetrack, or ocsort")
    if mode == MODE_DETECTOR_TRACKER and tracker_name == TRACKER_NONE:
        raise ValueError(
            "detector_tracker mode requires --tracker-name simple_iou, bytetrack, or ocsort"
        )
    if tracker_name in {TRACKER_BYTETRACK, TRACKER_OCSORT}:
        raise RuntimeError(
            f"tracker_name={tracker_name} is optional and not available in the core package. "
            "Install an external tracker adapter such as boxmot/ByteTrack/OC-SORT and wire it "
            "to this builder, or use --tracker-name simple_iou."
        )
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if not 0 <= nms_threshold <= 1:
        raise ValueError("nms_threshold must be in [0, 1]")
    if min_box_area < 0:
        raise ValueError("min_box_area must be non-negative")
    if max_objects_per_frame <= 0:
        raise ValueError("max_objects_per_frame must be positive")
    if max_missing < 0:
        raise ValueError("max_missing must be non-negative")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    if center_distance_threshold < 0:
        raise ValueError("center_distance_threshold must be non-negative")


def create_detector(
    *,
    detector_name: str,
    detector_model: str | None,
    fixture_detections_path: Path | None,
    confidence_threshold: float,
    nms_threshold: float,
    device: str | None,
) -> DetectorBackend:
    """Create the requested detector backend."""

    if detector_name == DETECTOR_FIXTURE_JSONL:
        if fixture_detections_path is None:
            raise RuntimeError(
                "detector_name=fixture_jsonl requires --fixture-detections pointing to "
                "a JSONL detector dump."
            )
        return FixtureJsonlDetector(fixture_detections_path)
    if detector_name == DETECTOR_ULTRALYTICS:
        return UltralyticsDetector(
            model_path=detector_model,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            device=device,
        )
    raise ValueError(f"Unsupported detector backend: {detector_name}")


def build_detector_only_records(
    frame_refs: list[FrameRef],
    *,
    detector: DetectorBackend,
    class_map: dict[str, str],
    confidence_threshold: float,
    nms_threshold: float,
    min_box_area: int,
    max_objects_per_frame: int,
    processed_frame_index_base: int,
) -> list[dict[str, Any]]:
    """Build detector-only records with per-frame temporary track ids."""

    records = []
    for ref in frame_refs:
        image = load_color_image(ref.path)
        height, width = image.shape[:2]
        detections = prepare_detections(
            detector.detect(ref, image),
            image_size=(height, width),
            class_map=class_map,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            min_box_area=min_box_area,
            max_objects_per_frame=max_objects_per_frame,
        )
        objects = [
            detection_to_object_record(
                detection,
                track_id=f"{ref.frame_idx}:{index}",
                width=width,
                height=height,
                velocity_xy_norm=(0.0, 0.0),
                age=1,
                missing=0,
                is_interpolated=False,
            )
            for index, detection in enumerate(detections, start=1)
        ]
        records.append(
            make_metadata_frame_record(
                video_id=ref.video_id,
                frame_idx=ref.frame_idx,
                image_size=(height, width),
                objects=objects,
                cache_source=CACHE_SOURCE_DETECTOR_ONLY,
                detector_name=detector.name,
                tracker_name=TRACKER_NONE,
                processed_frame_index_base=processed_frame_index_base,
            )
        )
    return records


def build_detector_tracker_records(
    frame_refs: list[FrameRef],
    *,
    detector: DetectorBackend,
    tracker_name: str,
    class_map: dict[str, str],
    confidence_threshold: float,
    nms_threshold: float,
    min_box_area: int,
    max_objects_per_frame: int,
    max_missing: int,
    iou_threshold: float,
    center_distance_threshold: float,
    processed_frame_index_base: int,
) -> list[dict[str, Any]]:
    """Build detector+tracker records with stable ids from simple association."""

    if tracker_name != TRACKER_SIMPLE_IOU:
        raise RuntimeError(
            f"tracker_name={tracker_name} is not available in this build. "
            "Use --tracker-name simple_iou or install/wire an external tracker adapter."
        )
    grouped = group_frame_refs_by_video(frame_refs)
    records = []
    next_track_id = 1
    for video_id, refs in grouped.items():
        active_tracks: list[ActiveTrack] = []
        for ref in refs:
            image = load_color_image(ref.path)
            height, width = image.shape[:2]
            detections = prepare_detections(
                detector.detect(ref, image),
                image_size=(height, width),
                class_map=class_map,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                min_box_area=min_box_area,
                max_objects_per_frame=max_objects_per_frame,
            )
            assignments, active_tracks, next_track_id = assign_detections_to_tracks(
                detections,
                active_tracks,
                next_track_id=next_track_id,
                image_size=(height, width),
                max_missing=max_missing,
                iou_threshold=iou_threshold,
                center_distance_threshold=center_distance_threshold,
            )
            objects = [
                detection_to_object_record(
                    detection,
                    track_id=track.track_id,
                    width=width,
                    height=height,
                    velocity_xy_norm=velocity_xy_norm,
                    age=track.age,
                    missing=0,
                    is_interpolated=False,
                )
                for detection, track, velocity_xy_norm in assignments
            ]
            records.append(
                make_metadata_frame_record(
                    video_id=video_id,
                    frame_idx=ref.frame_idx,
                    image_size=(height, width),
                    objects=objects,
                    cache_source=CACHE_SOURCE_DETECTOR_TRACKER,
                    detector_name=detector.name,
                    tracker_name=tracker_name,
                    processed_frame_index_base=processed_frame_index_base,
                )
            )
    return records


def build_oracle_mask_records(
    frame_refs: list[FrameRef],
    *,
    mask_root: Path | None,
    class_map: dict[str, str],
    min_box_area: int,
    max_objects_per_frame: int,
    processed_frame_index_base: int,
) -> list[dict[str, Any]]:
    """Build analysis-only records from anomaly pixel masks."""

    if mask_root is None:
        raise RuntimeError("oracle_mask mode requires --mask-root")
    records = []
    for ref in frame_refs:
        image = load_color_image(ref.path)
        height, width = image.shape[:2]
        detections = prepare_detections(
            detections_from_oracle_mask(ref, mask_root=mask_root, image_size=(height, width)),
            image_size=(height, width),
            class_map=class_map,
            confidence_threshold=0.0,
            nms_threshold=1.0,
            min_box_area=min_box_area,
            max_objects_per_frame=max_objects_per_frame,
        )
        objects = [
            detection_to_object_record(
                detection,
                track_id=f"oracle:{ref.frame_idx}:{index}",
                width=width,
                height=height,
                velocity_xy_norm=(0.0, 0.0),
                age=1,
                missing=0,
                is_interpolated=False,
            )
            for index, detection in enumerate(detections, start=1)
        ]
        record = make_metadata_frame_record(
            video_id=ref.video_id,
            frame_idx=ref.frame_idx,
            image_size=(height, width),
            objects=objects,
            cache_source=CACHE_SOURCE_ORACLE_MASK,
            detector_name=MODE_ORACLE_MASK,
            tracker_name=TRACKER_NONE,
            processed_frame_index_base=processed_frame_index_base,
        )
        record["analysis_only"] = True
        records.append(record)
    return records


def prepare_detections(
    detections: list[Detection],
    *,
    image_size: tuple[int, int],
    class_map: dict[str, str],
    confidence_threshold: float,
    nms_threshold: float,
    min_box_area: int,
    max_objects_per_frame: int,
) -> list[PreparedDetection]:
    """Clip, filter, remap, NMS, and cap detections for one frame."""

    height, width = image_size
    prepared = []
    for detection in detections:
        if detection.score < confidence_threshold:
            continue
        bbox = clip_bbox(detection.bbox_xyxy, width=width, height=height)
        if bbox_area(bbox) < min_box_area:
            continue
        if not is_valid_bbox(bbox, width=width, height=height):
            continue
        prepared.append(
            PreparedDetection(
                bbox_xyxy=bbox,
                score=float(detection.score),
                raw_class_id=int(detection.class_id),
                raw_class_name=str(detection.class_name),
                coarse_class_name=remap_class_name(detection.class_name, class_map),
            )
        )
    kept_indices = nms_indices(prepared, threshold=nms_threshold)
    kept = [prepared[index] for index in kept_indices]
    kept.sort(key=lambda item: item.score, reverse=True)
    return kept[:max_objects_per_frame]


def assign_detections_to_tracks(
    detections: list[PreparedDetection],
    active_tracks: list[ActiveTrack],
    *,
    next_track_id: int,
    image_size: tuple[int, int],
    max_missing: int,
    iou_threshold: float,
    center_distance_threshold: float,
) -> tuple[
    list[tuple[PreparedDetection, ActiveTrack, tuple[float, float]]], list[ActiveTrack], int
]:
    """Greedily associate prepared detections to active tracks."""

    height, width = image_size
    assignments: list[tuple[PreparedDetection, ActiveTrack, tuple[float, float]]] = []
    unmatched_tracks = active_tracks[:]
    updated_tracks: list[ActiveTrack] = []

    for detection in detections:
        match_index = best_detection_track_match(
            detection,
            unmatched_tracks,
            image_size=(height, width),
            iou_threshold=iou_threshold,
            center_distance_threshold=center_distance_threshold,
        )
        if match_index is None:
            track = ActiveTrack(
                track_id=str(next_track_id),
                bbox_xyxy=detection.bbox_xyxy,
                center_xy=detection.center_xy,
                age=1,
                missing=0,
            )
            velocity_xy_norm = (0.0, 0.0)
            next_track_id += 1
        else:
            previous = unmatched_tracks.pop(match_index)
            velocity_xy_norm = velocity_from_centers(
                previous.center_xy,
                detection.center_xy,
                width=width,
                height=height,
            )
            track = ActiveTrack(
                track_id=previous.track_id,
                bbox_xyxy=detection.bbox_xyxy,
                center_xy=detection.center_xy,
                age=previous.age + 1,
                missing=0,
            )
        assignments.append((detection, track, velocity_xy_norm))
        updated_tracks.append(track)

    for track in unmatched_tracks:
        if track.missing + 1 <= max_missing:
            track.missing += 1
            updated_tracks.append(track)
    return assignments, updated_tracks, next_track_id


def best_detection_track_match(
    detection: PreparedDetection,
    tracks: list[ActiveTrack],
    *,
    image_size: tuple[int, int],
    iou_threshold: float,
    center_distance_threshold: float,
) -> int | None:
    """Return the best active-track index for one detection."""

    height, width = image_size
    diagonal = math.sqrt(width * width + height * height)
    best_index = None
    best_score = -1.0
    for index, track in enumerate(tracks):
        current_iou = bbox_iou(detection.bbox_xyxy, track.bbox_xyxy)
        center_dist = normalized_center_distance(detection.center_xy, track.center_xy, diagonal)
        if current_iou < iou_threshold and center_dist > center_distance_threshold:
            continue
        score = current_iou - center_dist
        if score > best_score:
            best_index = index
            best_score = score
    return best_index


def detection_to_object_record(
    detection: PreparedDetection,
    *,
    track_id: str,
    width: int,
    height: int,
    velocity_xy_norm: tuple[float, float],
    age: int,
    missing: int,
    is_interpolated: bool,
) -> dict[str, Any]:
    """Serialize a prepared detection as one raw-cache object record."""

    record = make_object_record(
        track_id=track_id,
        bbox_xyxy=normalize_bbox(detection.bbox_xyxy, width=width, height=height),
        score=detection.score,
        class_id=detection.coarse_class_id,
        class_name=detection.coarse_class_name,
        velocity_xy_norm=velocity_xy_norm,
        age=age,
        is_interpolated=is_interpolated,
    )
    record["missing"] = int(missing)
    record["raw_class_id"] = int(detection.raw_class_id)
    record["raw_class_name"] = detection.raw_class_name
    return record


def make_metadata_frame_record(
    *,
    video_id: str,
    frame_idx: int,
    image_size: tuple[int, int],
    objects: list[dict[str, Any]],
    cache_source: str,
    detector_name: str,
    tracker_name: str,
    processed_frame_index_base: int,
) -> dict[str, Any]:
    """Return one frame record with Phase 1.5 cache metadata."""

    record = make_frame_record(
        video_id=video_id,
        frame_idx=frame_idx,
        image_size=image_size,
        objects=objects,
        cache_source=cache_source,
    )
    record["detector_name"] = detector_name
    record["tracker_name"] = tracker_name
    record["processed_frame_index_base"] = int(processed_frame_index_base)
    return record


def load_fixture_detections(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Load fixture/external detector JSONL records keyed by video/frame."""

    detections_by_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for line_number, record in enumerate(read_jsonl(path), start=1):
        if not isinstance(record, dict):
            raise ValueError(f"fixture detector record {path}:{line_number} must be an object")
        video_id = require_str(record, "video_id")
        frame_idx = require_int(record, "frame_idx")
        raw_detections = record.get("detections", record.get("objects", []))
        if not isinstance(raw_detections, list):
            raise ValueError(
                f"fixture detector record {path}:{line_number} detections must be a list"
            )
        detections_by_frame[(video_id, frame_idx)] = list(raw_detections)
    return detections_by_frame


def parse_fixture_detection(raw: dict[str, Any], *, width: int, height: int) -> Detection:
    """Parse one fixture detector object in pixel or normalized coordinates."""

    if not isinstance(raw, dict):
        raise ValueError("fixture detection must be an object")
    if "bbox_xyxy" in raw:
        bbox = parse_pixel_bbox(raw["bbox_xyxy"])
    elif "bbox_xyxy_norm" in raw:
        bbox = normalized_bbox_to_pixels(raw["bbox_xyxy_norm"], width=width, height=height)
    else:
        raise ValueError("fixture detection requires bbox_xyxy or bbox_xyxy_norm")
    return Detection(
        bbox_xyxy=bbox,
        score=float(raw.get("score", 1.0)),
        class_id=int(raw.get("class_id", -1)),
        class_name=str(raw.get("class_name", "unknown")),
    )


def detections_from_oracle_mask(
    ref: FrameRef,
    *,
    mask_root: Path,
    image_size: tuple[int, int],
) -> list[Detection]:
    """Convert one GT mask into connected-component detections."""

    mask_path = find_mask_path(mask_root, ref)
    if mask_path is None:
        return []
    mask = load_mask(mask_path)
    height, width = image_size
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    binary = (mask > 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    detections = []
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
            min_area=1,
            max_area=None,
        ):
            continue
        detections.append(
            Detection(
                bbox_xyxy=(x, y, x + box_w, y + box_h),
                score=1.0,
                class_id=COARSE_CLASS_IDS["unknown"],
                class_name="unknown",
            )
        )
    return detections


def find_mask_path(mask_root: Path, ref: FrameRef) -> Path | None:
    """Find a mask file matching one processed frame reference."""

    candidates = []
    video_dir = mask_root / ref.video_id
    for stem in (ref.path.stem, f"{ref.frame_idx:06d}", str(ref.frame_idx)):
        for suffix in (".png", ".jpg", ".jpeg", ".npy"):
            candidates.append(video_dir / f"{stem}{suffix}")
            candidates.append(mask_root / f"{ref.video_id}_{stem}{suffix}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_mask(path: Path) -> np.ndarray:
    """Load a mask image or numpy array as a 2D array."""

    if path.suffix.lower() == ".npy":
        value = np.load(path)
        if value.ndim == 3:
            value = value.max(axis=2)
        return value.astype(np.uint8)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read oracle mask: {path}")
    return mask


def load_color_image(path: Path) -> np.ndarray:
    """Load one processed frame as BGR uint8."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read frame image: {path}")
    return image


def clip_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Clip a pixel bbox to image bounds."""

    x1, y1, x2, y2 = bbox_xyxy
    return (
        int(max(0, min(width, x1))),
        int(max(0, min(height, y1))),
        int(max(0, min(width, x2))),
        int(max(0, min(height, y2))),
    )


def is_valid_bbox(bbox_xyxy: tuple[int, int, int, int], *, width: int, height: int) -> bool:
    """Return whether a clipped bbox is valid for the image."""

    x1, y1, x2, y2 = bbox_xyxy
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def bbox_area(bbox_xyxy: tuple[int, int, int, int]) -> int:
    """Return pixel bbox area."""

    x1, y1, x2, y2 = bbox_xyxy
    return max(0, x2 - x1) * max(0, y2 - y1)


def nms_indices(detections: list[PreparedDetection], *, threshold: float) -> list[int]:
    """Class-agnostic NMS over prepared detections."""

    if not detections:
        return []
    if threshold >= 1.0:
        return list(range(len(detections)))
    order = sorted(range(len(detections)), key=lambda index: detections[index].score, reverse=True)
    keep = []
    while order:
        current = order.pop(0)
        keep.append(current)
        order = [
            index
            for index in order
            if bbox_iou(detections[current].bbox_xyxy, detections[index].bbox_xyxy) <= threshold
        ]
    return keep


def parse_pixel_bbox(value: Any) -> tuple[int, int, int, int]:
    """Parse a pixel-space xyxy bbox."""

    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError("bbox_xyxy must be [x1, y1, x2, y2]")
    return tuple(int(round(float(item))) for item in value)  # type: ignore[return-value]


def normalized_bbox_to_pixels(value: Any, *, width: int, height: int) -> tuple[int, int, int, int]:
    """Convert normalized xyxy bbox to pixel coordinates."""

    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError("bbox_xyxy_norm must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (float(item) for item in value)
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def load_class_remap(path: Path | None) -> dict[str, str]:
    """Load optional raw-class to coarse VAD class mapping."""

    class_map = dict(DEFAULT_CLASS_REMAP)
    if path is None:
        return class_map
    with path.open("r", encoding="utf-8") as file:
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(file) or {}
        else:
            payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("class remap file must contain a mapping")
    for raw_name, coarse_name in payload.items():
        normalized_coarse = str(coarse_name).strip().lower()
        if normalized_coarse not in COARSE_CLASS_IDS:
            raise ValueError(
                "class remap values must be person, vehicle, two_wheeler, "
                f"carried_object, other_object, or unknown: {coarse_name}"
            )
        class_map[normalize_class_key(str(raw_name))] = normalized_coarse
    return class_map


def remap_class_name(class_name: str, class_map: dict[str, str]) -> str:
    """Map a detector class name to a coarse VAD class."""

    key = normalize_class_key(class_name)
    if not key:
        return "unknown"
    return class_map.get(key, "other_object")


def normalize_class_key(value: str) -> str:
    """Normalize a detector class-name key."""

    return value.strip().lower().replace("-", "_").replace(" ", "_")


def infer_processed_frame_index_base(frame_refs: Iterable[FrameRef]) -> int:
    """Infer frame-index base from processed frame indices."""

    min_idx = min(ref.frame_idx for ref in frame_refs)
    return 0 if min_idx == 0 else 1


def write_debug_visualizations(
    records: list[dict[str, Any]],
    *,
    frame_refs: list[FrameRef],
    output_dir: Path,
    max_frames: int,
) -> None:
    """Write a few frame overlays with boxes, track ids, and short trajectories."""

    output_dir.mkdir(parents=True, exist_ok=True)
    refs_by_key = {(ref.video_id, ref.frame_idx): ref for ref in frame_refs}
    track_history: dict[str, list[tuple[int, int]]] = {}
    written = 0
    for record in sorted(records, key=lambda item: (str(item["video_id"]), int(item["frame_idx"]))):
        if written >= max_frames:
            break
        ref = refs_by_key.get((str(record["video_id"]), int(record["frame_idx"])))
        if ref is None:
            continue
        image = load_color_image(ref.path)
        height, width = image.shape[:2]
        for obj in record["objects"]:
            x1, y1, x2, y2 = normalized_bbox_to_pixels(
                obj["bbox_xyxy_norm"], width=width, height=height
            )
            track_id = str(obj["track_id"])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            history = track_history.setdefault(track_id, [])
            history.append(center)
            history[:] = history[-20:]
            color = color_for_track(track_id)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                f"{track_id}:{obj.get('class_name', 'unknown')}",
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
            for first, second in zip(history, history[1:], strict=False):
                cv2.line(image, first, second, color, 1, cv2.LINE_AA)
        out_path = output_dir / f"{record['video_id']}_{int(record['frame_idx']):06d}.jpg"
        ok = cv2.imwrite(str(out_path), image)
        if not ok:
            raise RuntimeError(f"Failed to write debug visualization: {out_path}")
        written += 1


def color_for_track(track_id: str) -> tuple[int, int, int]:
    """Return a deterministic BGR color for a track id."""

    seed = abs(hash(track_id))
    return (50 + seed % 206, 50 + (seed // 7) % 206, 50 + (seed // 13) % 206)


def require_str(record: dict[str, Any], key: str) -> str:
    """Return a required nonempty string."""

    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a nonempty string")
    return value


def require_int(record: dict[str, Any], key: str) -> int:
    """Return a required integer."""

    value = record.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse detector/tracker cache CLI arguments."""

    parser = argparse.ArgumentParser(description="Build detector/tracker raw cache JSONL.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--feature-index", type=Path, default=None)
    parser.add_argument("--frames-root", type=Path, default=None)
    parser.add_argument("--output-track-cache", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--detector-name", choices=sorted(DETECTORS), default=DETECTOR_ULTRALYTICS)
    parser.add_argument("--detector-model", type=str, default=None)
    parser.add_argument("--fixture-detections", type=Path, default=None)
    parser.add_argument("--tracker-name", choices=sorted(TRACKERS), default=TRACKER_SIMPLE_IOU)
    parser.add_argument("--mask-root", type=Path, default=None)
    parser.add_argument("--class-map", type=Path, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.7)
    parser.add_argument("--min-box-area", type=int, default=16)
    parser.add_argument("--max-objects-per-frame", type=int, default=100)
    parser.add_argument("--max-missing", type=int, default=2)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--center-distance-threshold", type=float, default=0.2)
    parser.add_argument("--processed-frame-index-base", type=int, default=None)
    parser.add_argument("--debug-output-dir", type=Path, default=None)
    parser.add_argument("--debug-max-frames", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""

    args = parse_args(argv)
    records = build_detector_track_cache(
        output_track_cache=args.output_track_cache,
        mode=args.mode,
        config_path=args.config,
        feature_index_path=args.feature_index,
        frames_root=args.frames_root,
        detector_name=args.detector_name,
        detector_model=args.detector_model,
        fixture_detections_path=args.fixture_detections,
        tracker_name=args.tracker_name,
        mask_root=args.mask_root,
        class_map_path=args.class_map,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
        min_box_area=args.min_box_area,
        max_objects_per_frame=args.max_objects_per_frame,
        max_missing=args.max_missing,
        iou_threshold=args.iou_threshold,
        center_distance_threshold=args.center_distance_threshold,
        processed_frame_index_base=args.processed_frame_index_base,
        debug_output_dir=args.debug_output_dir,
        debug_max_frames=args.debug_max_frames,
        device=args.device,
    )
    print(f"wrote {len(records)} detector/tracker cache records -> {args.output_track_cache}")


if __name__ == "__main__":
    main(sys.argv[1:])
