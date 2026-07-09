import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from src.pipelines.detector_track_cache_builder import (
    CACHE_SOURCE_DETECTOR_ONLY,
    CACHE_SOURCE_DETECTOR_TRACKER,
    DETECTOR_FIXTURE_JSONL,
    MODE_DETECTOR_ONLY,
    MODE_DETECTOR_TRACKER,
    TRACKER_BYTETRACK,
    TRACKER_SIMPLE_IOU,
    build_detector_track_cache,
)
from src.pipelines.feature_eng_pipeline import read_jsonl, write_jsonl
from src.pipelines.object_track_pipeline import (
    build_track_feature_index,
    load_raw_track_cache,
    load_track_feature_payload,
    parse_track_frame,
)


class DetectorTrackCacheBuilderTests(unittest.TestCase):
    def test_detector_only_cache_has_valid_schema_and_debug_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames_root, frame_paths = make_frames_root(root, count=2)
            feature_index = make_feature_index(root, frame_paths)
            fixture = root / "detections.jsonl"
            write_fixture_detections(
                fixture,
                {
                    1: [
                        {
                            "bbox_xyxy": [10, 12, 26, 32],
                            "score": 0.91,
                            "class_id": 0,
                            "class_name": "person",
                        }
                    ],
                    2: [],
                },
            )
            output = root / "raw_track_cache.jsonl"
            debug_dir = root / "debug"

            records = build_detector_track_cache(
                output_track_cache=output,
                mode=MODE_DETECTOR_ONLY,
                feature_index_path=feature_index,
                detector_name=DETECTOR_FIXTURE_JSONL,
                fixture_detections_path=fixture,
                confidence_threshold=0.2,
                min_box_area=4,
                debug_output_dir=debug_dir,
                debug_max_frames=1,
            )
            written = read_jsonl(output)
            frame = parse_track_frame(records[0])

            self.assertEqual(records, written)
            self.assertEqual(records[0]["cache_source"], CACHE_SOURCE_DETECTOR_ONLY)
            self.assertEqual(records[0]["detector_name"], DETECTOR_FIXTURE_JSONL)
            self.assertEqual(records[0]["tracker_name"], "none")
            self.assertEqual(records[0]["processed_frame_index_base"], 1)
            self.assertEqual(frame.objects[0].class_name, "person")
            self.assertTrue(any(debug_dir.glob("*.jpg")))

    def test_detector_tracker_keeps_stable_ids_on_moving_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames_root, _ = make_frames_root(root, count=3)
            fixture = root / "moving_detections.jsonl"
            write_fixture_detections(
                fixture,
                {
                    1: [person_box(10, 10, 24, 28)],
                    2: [person_box(12, 10, 26, 28)],
                    3: [person_box(14, 10, 28, 28)],
                },
            )
            output = root / "raw_track_cache.jsonl"

            records = build_detector_track_cache(
                output_track_cache=output,
                mode=MODE_DETECTOR_TRACKER,
                frames_root=frames_root,
                detector_name=DETECTOR_FIXTURE_JSONL,
                fixture_detections_path=fixture,
                tracker_name=TRACKER_SIMPLE_IOU,
                confidence_threshold=0.1,
                min_box_area=4,
                iou_threshold=0.1,
                center_distance_threshold=0.5,
            )

        track_ids = [record["objects"][0]["track_id"] for record in records]
        ages = [record["objects"][0]["age"] for record in records]
        self.assertEqual(len(set(track_ids)), 1)
        self.assertEqual(ages, [1, 2, 3])
        self.assertEqual(records[0]["cache_source"], CACHE_SOURCE_DETECTOR_TRACKER)
        self.assertEqual(records[0]["tracker_name"], TRACKER_SIMPLE_IOU)
        self.assertNotEqual(records[1]["objects"][0]["velocity_xy_norm"], [0.0, 0.0])

    def test_empty_detection_frames_are_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames_root, _ = make_frames_root(root, count=2)
            fixture = root / "empty_detections.jsonl"
            write_fixture_detections(fixture, {1: [], 2: []})
            output = root / "empty_cache.jsonl"

            records = build_detector_track_cache(
                output_track_cache=output,
                mode=MODE_DETECTOR_ONLY,
                frames_root=frames_root,
                detector_name=DETECTOR_FIXTURE_JSONL,
                fixture_detections_path=fixture,
            )

        self.assertEqual([record["objects"] for record in records], [[], []])

    def test_class_remapping_to_coarse_vad_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames_root, _ = make_frames_root(root, count=1)
            fixture = root / "classes.jsonl"
            write_fixture_detections(
                fixture,
                {
                    1: [
                        class_box("car", 2, 2, 10, 10, 2),
                        class_box("bicycle", 12, 2, 20, 10, 1),
                        class_box("backpack", 22, 2, 30, 10, 24),
                        class_box("dog", 32, 2, 40, 10, 16),
                        class_box("unknown", 42, 2, 50, 10, -1),
                    ]
                },
            )
            output = root / "classes_cache.jsonl"

            records = build_detector_track_cache(
                output_track_cache=output,
                mode=MODE_DETECTOR_ONLY,
                frames_root=frames_root,
                detector_name=DETECTOR_FIXTURE_JSONL,
                fixture_detections_path=fixture,
                confidence_threshold=0.1,
                min_box_area=4,
            )

        class_names = [obj["class_name"] for obj in records[0]["objects"]]
        self.assertIn("vehicle", class_names)
        self.assertIn("two_wheeler", class_names)
        self.assertIn("carried_object", class_names)
        self.assertIn("other_object", class_names)
        self.assertIn("unknown", class_names)

    def test_generated_cache_can_be_consumed_by_phase1_feature_builder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames_root, frame_paths = make_frames_root(root, count=3)
            feature_index = make_feature_index(root, frame_paths, include_z=True)
            fixture = root / "detections.jsonl"
            write_fixture_detections(
                fixture,
                {
                    1: [person_box(8, 8, 24, 24)],
                    2: [person_box(10, 8, 26, 24)],
                    3: [person_box(12, 8, 28, 24)],
                },
            )
            track_cache = root / "raw_track_cache.jsonl"
            augmented_index = root / "features" / "test_feature_index_tracks.jsonl"

            build_detector_track_cache(
                output_track_cache=track_cache,
                mode=MODE_DETECTOR_TRACKER,
                frames_root=frames_root,
                detector_name=DETECTOR_FIXTURE_JSONL,
                fixture_detections_path=fixture,
                tracker_name=TRACKER_SIMPLE_IOU,
                confidence_threshold=0.1,
                min_box_area=4,
            )
            cache = load_raw_track_cache(track_cache)
            records = build_track_feature_index(
                feature_index_path=feature_index,
                track_cache_path=track_cache,
                output_feature_index_path=augmented_index,
                z_patch_size=2,
            )
            payload = load_track_feature_payload(Path(records[0]["track_feature_path"]))

            self.assertIn(("video", 3), cache)
            self.assertTrue(augmented_index.is_file())
            self.assertGreater(float(payload["future_grid"].sum()), 0.0)

    def test_optional_bytetrack_fails_with_clear_message(self):
        with self.assertRaisesRegex(RuntimeError, "optional.*core package.*simple_iou"):
            build_detector_track_cache(
                output_track_cache=Path("unused.jsonl"),
                mode=MODE_DETECTOR_TRACKER,
                detector_name=DETECTOR_FIXTURE_JSONL,
                fixture_detections_path=Path("unused.jsonl"),
                tracker_name=TRACKER_BYTETRACK,
            )


def make_frames_root(
    root: Path,
    *,
    count: int,
) -> tuple[Path, list[Path]]:
    frames_root = root / "frames" / "test"
    video_dir = frames_root / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for index in range(1, count + 1):
        path = video_dir / f"{index:06d}.jpg"
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        ok = cv2.imwrite(str(path), image)
        if not ok:
            raise RuntimeError(f"failed to write test frame: {path}")
        frame_paths.append(path)
    return frames_root, frame_paths


def write_fixture_detections(path: Path, detections_by_frame: dict[int, list[dict[str, object]]]):
    records = [
        {"video_id": "video", "frame_idx": frame_idx, "detections": detections}
        for frame_idx, detections in sorted(detections_by_frame.items())
    ]
    write_jsonl(path, records)


def person_box(x1: int, y1: int, x2: int, y2: int) -> dict[str, object]:
    return class_box("person", x1, y1, x2, y2, 0)


def class_box(
    class_name: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    class_id: int,
) -> dict[str, object]:
    return {
        "bbox_xyxy": [x1, y1, x2, y2],
        "score": 0.9,
        "class_id": class_id,
        "class_name": class_name,
    }


def make_feature_index(
    root: Path,
    frame_paths: list[Path],
    *,
    include_z: bool = False,
) -> Path:
    features = root / "features"
    z_path = features / "z" / "test" / "sample.pt"
    if include_z:
        z_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"sample_id": "sample", "z": torch.zeros(2, 2, 4, 4)}, z_path)
    record = {
        "sample_id": "sample",
        "video_id": "video",
        "split": "test",
        "context_frames": [1, 2],
        "future_frames": [3],
        "context_frame_paths": [str(frame_paths[0]), str(frame_paths[1])],
        "future_frame_paths": [
            str(frame_paths[2]) if len(frame_paths) > 2 else str(frame_paths[-1])
        ],
        "future_frame_labels": [0],
        "future_label": 0,
    }
    if include_z:
        record["z_path"] = str(z_path)
    feature_index = features / "test_feature_index.jsonl"
    write_jsonl(feature_index, [record])
    return feature_index


if __name__ == "__main__":
    unittest.main()
