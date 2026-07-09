from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.pipelines.feature_eng_pipeline import read_jsonl, write_jsonl
from src.pipelines.object_track_vad_pipeline import (
    SCORE_CLASS_SCENE_MEMORY,
    aggregate_frame_scores_from_tracks,
    build_memory_bank,
    build_track_windows,
    evaluate_object_track_vad,
    load_track_state_cache,
    score_track_windows,
)


class ObjectTrackVadPipelineTests(unittest.TestCase):
    def test_synthetic_moving_track_produces_stable_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracks.jsonl"
            write_track_cache(path, moving_records("01_001", "normal", velocity=(0.05, 0.0), count=4))

            cache = load_track_state_cache(path)
            windows = build_track_windows(cache.observations, context_length=3)

            self.assertEqual(len(cache.observations), 4)
            self.assertEqual(len(windows), 4)
            last = cache.observations[-1]
            self.assertAlmostEqual(last.state[4], 0.05, places=6)
            self.assertAlmostEqual(last.state[6], 0.05, places=6)
            self.assertEqual(windows[0].mask, (False, False, True))
            self.assertEqual(windows[-1].mask, (True, True, True))

    def test_memory_bank_builds_from_normal_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_tracks.jsonl"
            write_track_cache(path, moving_records("01_001", "normal", velocity=(0.02, 0.0), count=5))

            cache = load_track_state_cache(path)
            windows = build_track_windows(cache.observations, context_length=3)
            memory = build_memory_bank(
                windows,
                context_length=3,
                max_prototypes=8,
                prototype_method="random",
                seed=3,
            )

            self.assertIn("__global__", memory.groups)
            self.assertIn("scene:scene_01", memory.groups)
            self.assertIn("class:0", memory.groups)
            self.assertGreaterEqual(memory.groups["__global__"].shape[0], 1)

    def test_anomalous_fast_track_scores_higher_than_normal_speed_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "train_tracks.jsonl"
            test_path = Path(tmp) / "test_tracks.jsonl"
            write_track_cache(
                train_path,
                moving_records("01_001", "normal", velocity=(0.02, 0.0), count=6),
            )
            records = []
            records.extend(moving_records("01_002", "normal_test", velocity=(0.02, 0.0), count=6))
            records.extend(moving_records("01_003", "fast_test", velocity=(0.12, 0.0), count=6))
            write_track_cache(test_path, records)

            train_cache = load_track_state_cache(train_path)
            memory = build_memory_bank(
                build_track_windows(train_cache.observations, context_length=3),
                context_length=3,
                max_prototypes=16,
                prototype_method="random",
            )
            test_cache = load_track_state_cache(test_path)
            scores = score_track_windows(
                build_track_windows(test_cache.observations, context_length=3),
                memory,
            )

            normal_score = last_score_for_track(scores, "normal_test", SCORE_CLASS_SCENE_MEMORY)
            fast_score = last_score_for_track(scores, "fast_test", SCORE_CLASS_SCENE_MEMORY)
            self.assertGreater(fast_score, normal_score)

    def test_empty_frames_are_handled_safely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracks.jsonl"
            records = [
                frame_record("01_001", 0, []),
                *moving_records("01_001", "track", velocity=(0.01, 0.0), count=2, start_frame=1),
            ]
            write_track_cache(path, records)
            cache = load_track_state_cache(path)
            windows = build_track_windows(cache.observations, context_length=2)
            scores = score_track_windows(
                windows,
                build_memory_bank(
                    windows,
                    context_length=2,
                    max_prototypes=4,
                    prototype_method="random",
                ),
            )

            frame_scores = aggregate_frame_scores_from_tracks(
                frames=cache.frames,
                track_score_records=scores,
                labels={("01_001", 0): 0, ("01_001", 1): 0},
                primary_score_mode=SCORE_CLASS_SCENE_MEMORY,
                frame_aggregation="topk_mean",
                frame_top_k=2,
            )

            self.assertEqual(frame_scores[0]["frame_idx"], 0)
            self.assertEqual(frame_scores[0]["num_tracks"], 0)
            self.assertEqual(frame_scores[0]["score"], 0.0)

    def test_frame_score_aggregation_uses_topk_mean(self) -> None:
        records = [
            {"video_id": "v", "frame_idx": 0, "track_id": "a", "nearest_memory_distance": 1.0, "trajectory_velocity_distance": 0.5, "class_scene_memory_distance": 1.0},
            {"video_id": "v", "frame_idx": 0, "track_id": "b", "nearest_memory_distance": 3.0, "trajectory_velocity_distance": 0.5, "class_scene_memory_distance": 3.0},
            {"video_id": "v", "frame_idx": 0, "track_id": "c", "nearest_memory_distance": 5.0, "trajectory_velocity_distance": 0.5, "class_scene_memory_distance": 5.0},
        ]
        frame_scores = aggregate_frame_scores_from_tracks(
            frames=(),
            track_score_records=records,
            labels={("v", 0): 1},
            primary_score_mode=SCORE_CLASS_SCENE_MEMORY,
            frame_aggregation="topk_mean",
            frame_top_k=2,
        )

        self.assertEqual(len(frame_scores), 1)
        self.assertEqual(frame_scores[0]["label"], 1)
        self.assertAlmostEqual(frame_scores[0]["score"], 4.0)
        self.assertAlmostEqual(frame_scores[0]["topk_track_frame_score_raw_score"], 4.0)

    def test_evaluate_object_track_vad_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train_tracks.jsonl"
            test_path = root / "test_tracks.jsonl"
            labels_path = root / "labels.csv"
            output_dir = root / "out"
            write_track_cache(
                train_path,
                moving_records("01_001", "normal", velocity=(0.02, 0.0), count=6),
            )
            records = []
            records.extend(moving_records("01_002", "normal_test", velocity=(0.02, 0.0), count=6))
            records.extend(moving_records("01_003", "fast_test", velocity=(0.12, 0.0), count=6))
            write_track_cache(test_path, records)
            write_labels(labels_path, [("01_002", idx, 0) for idx in range(6)])
            append_labels(labels_path, [("01_003", idx, 1) for idx in range(6)])

            evaluate_object_track_vad(
                train_track_cache=train_path,
                test_track_cache=test_path,
                output_dir=output_dir,
                labels_path=labels_path,
                context_length=3,
                max_prototypes=16,
                prototype_method="random",
                primary_score_mode=SCORE_CLASS_SCENE_MEMORY,
                frame_aggregation="topk_mean",
                frame_top_k=2,
            )

            self.assertTrue((output_dir / "memory_bank.pt").is_file())
            self.assertTrue((output_dir / "per_track_scores.csv").is_file())
            frame_scores = read_jsonl(output_dir / "frame_scores.jsonl")
            metrics = read_json(output_dir / "metrics.json")
            self.assertTrue(frame_scores)
            self.assertIn("score", frame_scores[0])
            self.assertIn("roc_auc", metrics)
            self.assertIn("average_precision", metrics)
            self.assertIsNotNone(metrics["roc_auc"])
            self.assertIsNotNone(metrics["average_precision"])


def moving_records(
    video_id: str,
    track_id: str,
    *,
    velocity: tuple[float, float],
    count: int,
    start_frame: int = 0,
) -> list[dict]:
    records = []
    vx, vy = velocity
    for offset in range(count):
        frame_idx = start_frame + offset
        cx = 0.2 + vx * offset
        cy = 0.5 + vy * offset
        bbox = [cx - 0.04, cy - 0.05, cx + 0.04, cy + 0.05]
        records.append(
            frame_record(
                video_id,
                frame_idx,
                [
                    {
                        "track_id": track_id,
                        "bbox_xyxy_norm": bbox,
                        "score": 0.9,
                        "class_id": 0,
                        "class_name": "person",
                        "velocity_xy_norm": [vx, vy],
                        "age": offset + 1,
                        "missing": 0,
                        "is_interpolated": False,
                    }
                ],
            )
        )
    return records


def frame_record(video_id: str, frame_idx: int, objects: list[dict]) -> dict:
    return {
        "video_id": video_id,
        "scene_id": "scene_01",
        "frame_idx": frame_idx,
        "image_size": [64, 64],
        "objects": objects,
        "cache_source": "test",
    }


def write_track_cache(path: Path, records: list[dict]) -> None:
    write_jsonl(path, records)


def write_labels(path: Path, rows: list[tuple[str, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["video_id", "frame_idx", "label"])
        writer.writeheader()
        for video_id, frame_idx, label in rows:
            writer.writerow({"video_id": video_id, "frame_idx": frame_idx, "label": label})


def append_labels(path: Path, rows: list[tuple[str, int, int]]) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["video_id", "frame_idx", "label"])
        for video_id, frame_idx, label in rows:
            writer.writerow({"video_id": video_id, "frame_idx": frame_idx, "label": label})


def last_score_for_track(records: list[dict], track_id: str, score_key: str) -> float:
    candidates = [record for record in records if record["track_id"] == track_id]
    self_record = sorted(candidates, key=lambda item: item["frame_idx"])[-1]
    return float(self_record[score_key])


def read_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text())


if __name__ == "__main__":
    unittest.main()
