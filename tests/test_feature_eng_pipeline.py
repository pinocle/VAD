import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from src.pipelines.feature_eng_pipeline import (
    TESTING_PHASE,
    TRAINING_PHASE,
    PreprocessConfig,
    SamplingConfig,
    context_future_windows,
    discover_video_records,
    load_config,
    read_jsonl,
    run_preprocess,
)


class FeatureEngineeringPipelineTests(unittest.TestCase):
    def test_load_config_accepts_ma_pdm_roots_and_auto_indexing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset": {
                            "name": "shanghaitech",
                            "raw_root": str(root / "raw"),
                            "processed_root": str(root / "processed"),
                            "test_labels_path": "",
                        },
                        "preprocess": {
                            "image_ext": "jpg",
                            "jpeg_quality": 95,
                            "frame_index_start": "auto",
                        },
                        "sampling": {
                            "context_frames": 2,
                            "future_frames": 1,
                            "gap": 0,
                            "stride_train": 1,
                            "stride_test": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.dataset_name, "shanghai")
        self.assertEqual(config.raw_dataset_root, root / "raw" / "shanghai")
        self.assertEqual(config.processed_dataset_root, root / "processed" / "shanghai")
        self.assertEqual(config.frame_index_start, 1)
        self.assertIsNone(config.test_labels_path)

    def test_discovers_only_ma_pdm_frame_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            create_frames(raw / "shanghai" / TRAINING_PHASE / "frames" / "01_001", count=2)
            create_frames(raw / "shanghai" / TESTING_PHASE / "frames" / "01_0001", count=2)
            masks = raw / "shanghai" / TESTING_PHASE / "test_frame_mask"
            masks.mkdir(parents=True)
            np.save(masks / "01_0001.npy", np.array([0, 1], dtype=np.uint8))

            records = discover_video_records(make_config(raw, root / "processed"), phases=None)

        self.assertEqual(
            [(record.video_id, record.phase, record.frame_dir.name) for record in records],
            [
                ("01_001", TRAINING_PHASE, "01_001"),
                ("01_0001", TESTING_PHASE, "01_0001"),
            ],
        )
        self.assertIsNone(records[0].labels)
        self.assertEqual(records[1].labels.tolist(), [0, 1])

    def test_preprocess_copies_frames_and_writes_minimal_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            create_frames(
                raw / "shanghai" / TRAINING_PHASE / "frames" / "01_001", count=5, ext="png"
            )
            create_frames(raw / "shanghai" / TESTING_PHASE / "frames" / "01_0001", count=5)
            masks = raw / "shanghai" / TESTING_PHASE / "test_frame_mask"
            masks.mkdir(parents=True)
            np.save(masks / "01_0001.npy", np.array([0, 0, 1, 1, 0], dtype=np.uint8))
            config = make_config(
                raw,
                root / "processed",
                sampling=SamplingConfig(2, 2, 1, 1, 1),
            )

            train_count, test_count = run_preprocess(config)
            manifest = read_jsonl(config.manifest_path)
            frame_labels = read_jsonl(config.frame_labels_path)
            train_samples = read_jsonl(config.train_samples_path)
            test_samples = read_jsonl(config.test_samples_path)
            layout = json.loads((config.metadata_root / "layout.json").read_text(encoding="utf-8"))

            processed = root / "processed" / "shanghai"
            self.assertTrue(
                (processed / TRAINING_PHASE / "frames" / "01_001" / "000001.jpg").is_file()
            )
            self.assertFalse(
                (processed / TRAINING_PHASE / "frames" / "01_001" / "000001.png").exists()
            )

        self.assertEqual((train_count, test_count), (1, 1))
        self.assertEqual(
            [(row["video_id"], row["phase"], row["num_frames"]) for row in manifest],
            [("01_001", TRAINING_PHASE, 5), ("01_0001", TESTING_PHASE, 5)],
        )
        self.assertEqual(frame_labels[-3]["label"], 1)
        self.assertEqual(train_samples[0]["context_frames"], [1, 2])
        self.assertEqual(train_samples[0]["future_frames"], [4, 5])
        self.assertEqual(test_samples[0]["future_frame_labels"], [1, 0])
        self.assertTrue(test_samples[0]["future_frame_paths"][0].endswith("000004.jpg"))
        self.assertEqual(layout["layout"], "ma_pdm")
        self.assertEqual(layout["sampling"]["stride_train"], 1)

    def test_preprocess_honors_split_files_and_zero_based_ucf_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            for phase in (TRAINING_PHASE, TESTING_PHASE):
                create_frames(raw / "ucf" / phase / "frames" / "keep", count=2)
                create_frames(raw / "ucf" / phase / "frames" / "drop", count=2)
            train_split = root / "train.json"
            test_split = root / "test.json"
            test_labels = root / "labels.json"
            train_split.write_text(json.dumps(["keep"]), encoding="utf-8")
            test_split.write_text(json.dumps({"keep": {}}), encoding="utf-8")
            test_labels.write_text(json.dumps({"keep": [0, 1]}), encoding="utf-8")
            config = make_config(
                raw,
                root / "processed",
                dataset_name="ucf",
                frame_index_start=0,
                train_split_path=train_split,
                test_split_path=test_split,
                test_labels_path=test_labels,
                sampling=SamplingConfig(1, 1, 0, 1, 1),
            )

            train_count, test_count = run_preprocess(config)
            test_samples = read_jsonl(config.test_samples_path)

            self.assertTrue((config.frames_root(TRAINING_PHASE) / "keep" / "000000.jpg").is_file())
            self.assertFalse((config.frames_root(TRAINING_PHASE) / "drop").exists())

        self.assertEqual((train_count, test_count), (1, 1))
        self.assertEqual(test_samples[0]["future_frames"], [1])
        self.assertEqual(test_samples[0]["future_label"], 1)

    def test_context_windows_match_ma_pdm_dense_scan(self) -> None:
        sampling = SamplingConfig(3, 2, 1, 1, 1)

        windows = list(context_future_windows(8, sampling, stride=1))

        self.assertEqual(
            windows,
            [
                ([0, 1, 2], [4, 5]),
                ([1, 2, 3], [5, 6]),
                ([2, 3, 4], [6, 7]),
            ],
        )


def make_config(
    raw_root: Path,
    processed_root: Path,
    *,
    dataset_name: str = "shanghai",
    frame_index_start: int = 1,
    train_split_path: Path | None = None,
    test_split_path: Path | None = None,
    test_labels_path: Path | None = None,
    sampling: SamplingConfig | None = None,
) -> PreprocessConfig:
    return PreprocessConfig(
        dataset_name=dataset_name,
        raw_root=raw_root,
        processed_root=processed_root,
        image_ext="jpg",
        jpeg_quality=95,
        frame_index_start=frame_index_start,
        overwrite=True,
        train_split_path=train_split_path,
        test_split_path=test_split_path,
        test_labels_path=test_labels_path,
        sampling=sampling or SamplingConfig(2, 1, 0, 1, 1),
    )


def create_frames(directory: Path, *, count: int, ext: str = "jpg") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in reversed(range(count)):
        Image.new("RGB", (8, 6), color=(index * 20, 40, 80)).save(directory / f"{index:03d}.{ext}")


if __name__ == "__main__":
    unittest.main()
