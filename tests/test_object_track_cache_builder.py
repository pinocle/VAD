import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from src.pipelines.feature_eng_pipeline import read_jsonl, write_jsonl
from src.pipelines.object_track_cache_builder import (
    MODE_MOTION_PROPOSAL,
    MODE_SYNTHETIC,
    build_pseudo_track_cache,
)
from src.pipelines.object_track_pipeline import (
    build_track_feature_index,
    load_raw_track_cache,
    load_track_feature_payload,
    parse_track_frame,
)


class ObjectTrackCacheBuilderTests(unittest.TestCase):
    def test_synthetic_cache_has_valid_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames_root, frame_paths = make_frames_root(root, count=3)
            feature_index = make_feature_index(root, frame_paths)
            output = root / "raw_track_cache.jsonl"

            records = build_pseudo_track_cache(
                output_track_cache=output,
                mode=MODE_SYNTHETIC,
                feature_index_path=feature_index,
            )
            written = read_jsonl(output)

        self.assertEqual(len(records), 3)
        self.assertEqual(records, written)
        self.assertEqual(records[0]["cache_source"], MODE_SYNTHETIC)
        frame = parse_track_frame(records[0])
        self.assertEqual(frame.video_id, "video")
        self.assertEqual(frame.image_size, (64, 64))
        self.assertEqual(records[0]["objects"][0]["class_name"], "synthetic_object")

    def test_empty_frames_produce_empty_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames_root, _ = make_frames_root(root, count=3)
            output = root / "motion_empty.jsonl"

            records = build_pseudo_track_cache(
                output_track_cache=output,
                mode=MODE_MOTION_PROPOSAL,
                frames_root=frames_root,
                min_area=8,
                diff_percentile=80.0,
            )

        self.assertTrue(records)
        self.assertTrue(all(record["objects"] == [] for record in records))

    def test_motion_proposal_detects_moving_square_and_keeps_track_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames_root, _ = make_frames_root(
                root,
                squares=[
                    None,
                    (8, 20, 20, 32),
                    (12, 20, 24, 32),
                    (16, 20, 28, 32),
                ],
            )
            output = root / "motion.jsonl"

            records = build_pseudo_track_cache(
                output_track_cache=output,
                mode=MODE_MOTION_PROPOSAL,
                frames_root=frames_root,
                min_area=8,
                diff_percentile=75.0,
                diff_std_k=0.5,
                max_missing=1,
                iou_threshold=0.1,
                center_distance_threshold=0.5,
            )
            object_records = [record for record in records if record["objects"]]

        self.assertGreaterEqual(len(object_records), 2)
        self.assertEqual(object_records[0]["cache_source"], MODE_MOTION_PROPOSAL)
        self.assertEqual(
            object_records[0]["objects"][0]["class_name"],
            "unknown_moving_object",
        )
        self.assertEqual(
            object_records[0]["objects"][0]["track_id"],
            object_records[1]["objects"][0]["track_id"],
        )
        self.assertNotEqual(object_records[1]["objects"][0]["velocity_xy_norm"], [0.0, 0.0])

    def test_bbox_values_are_normalized_and_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames_root, _ = make_frames_root(root, count=2)
            output = root / "synthetic.jsonl"

            records = build_pseudo_track_cache(
                output_track_cache=output,
                mode=MODE_SYNTHETIC,
                frames_root=frames_root,
            )

        bboxes = [obj["bbox_xyxy_norm"] for record in records for obj in record["objects"]]
        self.assertTrue(bboxes)
        for x1, y1, x2, y2 in bboxes:
            self.assertTrue(0.0 <= x1 < x2 <= 1.0)
            self.assertTrue(0.0 <= y1 < y2 <= 1.0)

    def test_output_can_be_consumed_by_phase1_loader_and_feature_builder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames_root, frame_paths = make_frames_root(root, count=3)
            feature_index = make_feature_index(root, frame_paths, include_z=True)
            track_cache = root / "raw_track_cache.jsonl"
            augmented_index = root / "features" / "test_feature_index_tracks.jsonl"

            build_pseudo_track_cache(
                output_track_cache=track_cache,
                mode=MODE_SYNTHETIC,
                frames_root=frames_root,
            )
            cache = load_raw_track_cache(track_cache)
            records = build_track_feature_index(
                feature_index_path=feature_index,
                track_cache_path=track_cache,
                output_feature_index_path=augmented_index,
                z_patch_size=2,
            )
            payload = load_track_feature_payload(Path(records[0]["track_feature_path"]))

            self.assertIn(("video", 1), cache)
            self.assertTrue(augmented_index.is_file())
            self.assertEqual(tuple(payload["future_grid"].shape[-2:]), (2, 2))


def make_frames_root(
    root: Path,
    *,
    count: int | None = None,
    squares: list[tuple[int, int, int, int] | None] | None = None,
) -> tuple[Path, list[Path]]:
    frames_root = root / "frames" / "test"
    video_dir = frames_root / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    if squares is None:
        squares = [None] * int(count or 0)
    frame_paths = []
    for index, square in enumerate(squares, start=1):
        path = video_dir / f"{index:06d}.jpg"
        write_frame(path, square=square)
        frame_paths.append(path)
    return frames_root, frame_paths


def write_frame(path: Path, *, square: tuple[int, int, int, int] | None = None) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    if square is not None:
        x1, y1, x2, y2 = square
        image[y1:y2, x1:x2] = 255
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"failed to write test frame: {path}")


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
        "future_frame_paths": [str(frame_paths[2])],
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
