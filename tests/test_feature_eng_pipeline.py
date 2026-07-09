import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from src.pipelines.feature_eng_pipeline import (
    FRAME_DIR_SOURCE,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VIDEO_SOURCE,
    FrameMap,
    PreparedVideo,
    PreprocessConfig,
    SamplingConfig,
    VideoRecord,
    build_sample_indices,
    build_video_records,
    context_future_windows,
    labels_for_record,
    list_image_files,
    load_config,
    load_frame_mask,
    materialize_frame_dir,
    materialize_frame_files,
    parse_fps,
    read_jsonl,
    scene_class,
)


class FeatureEngineeringPipelineTests(unittest.TestCase):
    def test_load_config_uses_native_fps_and_window_defaults(self):
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
                        },
                        "preprocess": {"fps": "native"},
                        "sampling": {
                            "context_frames": 32,
                            "future_frames": 8,
                            "context_sampling": "dense",
                            "context_span": 32,
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertIsNone(config.fps)
        self.assertEqual(config.image_ext, "jpg")
        self.assertEqual(config.num_workers, 8)
        self.assertEqual(config.image_backend, "auto")
        self.assertEqual(config.sampling.context_frames, 32)
        self.assertEqual(config.sampling.future_frames, 8)

    def test_load_config_validates_optimization_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            base_config = {
                "dataset": {
                    "name": "shanghaitech",
                    "raw_root": str(root / "raw"),
                    "processed_root": str(root / "processed"),
                },
                "preprocess": {
                    "num_workers": -1,
                    "image_backend": "auto",
                },
                "sampling": {
                    "context_frames": 32,
                    "future_frames": 8,
                    "context_sampling": "dense",
                    "context_span": 32,
                },
            }
            config_path.write_text(yaml.safe_dump(base_config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "num_workers"):
                load_config(config_path)

            base_config["preprocess"]["num_workers"] = 1
            base_config["preprocess"]["image_backend"] = "unknown"
            config_path.write_text(yaml.safe_dump(base_config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "image_backend"):
                load_config(config_path)

    def test_parse_fps_accepts_native_or_positive_integer_only(self):
        self.assertIsNone(parse_fps("native"))
        self.assertEqual(parse_fps("15"), 15)
        self.assertEqual(parse_fps(30), 30)
        with self.assertRaises(ValueError):
            parse_fps(0)
        with self.assertRaises(ValueError):
            parse_fps("target")

    def test_build_video_records_reads_shanghaitech_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            train = raw / "training" / "videos"
            test_frames = raw / "testing" / "frames" / "01_0014"
            masks = raw / "testing" / "test_frame_mask"
            train.mkdir(parents=True)
            test_frames.mkdir(parents=True)
            masks.mkdir(parents=True)
            (train / "01_001.avi").write_bytes(b"fake")
            np.save(masks / "01_0014.npy", np.array([0, 1, 0], dtype=np.uint8))
            config = PreprocessConfig(
                raw_root=raw,
                processed_root=root / "processed",
            )

            records = build_video_records(config)

        self.assertEqual([record.video_id for record in records], ["01_001", "01_0014"])
        self.assertEqual(records[0].split, TRAIN_SPLIT)
        self.assertEqual(records[1].split, TEST_SPLIT)
        self.assertTrue(records[1].is_anomaly)
        self.assertEqual(scene_class("01_0014"), "scene_01")

    def test_build_video_records_reads_avenue_layout(self):
        from scipy.io import savemat

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            train = raw / "training_videos"
            test = raw / "testing_videos"
            labels = raw / "ground_truth_demo" / "testing_label_mask"
            train.mkdir(parents=True)
            test.mkdir(parents=True)
            labels.mkdir(parents=True)
            (train / "01.avi").write_bytes(b"fake")
            (test / "01.avi").write_bytes(b"fake")
            cells = np.empty((2,), dtype=object)
            cells[0] = np.zeros((2, 2), dtype=np.uint8)
            cells[1] = np.ones((2, 2), dtype=np.uint8)
            savemat(labels / "1_label.mat", {"volLabel": cells})
            config = PreprocessConfig(
                raw_root=raw,
                processed_root=root / "processed",
                dataset_name="avenue",
                label_source="ground_truth_demo/testing_label_mask",
            )

            records = build_video_records(config)

        self.assertEqual([record.video_id for record in records], ["01", "01"])
        self.assertEqual(records[0].split, TRAIN_SPLIT)
        self.assertEqual(records[1].split, TEST_SPLIT)
        self.assertEqual(records[1].source_type, VIDEO_SOURCE)
        self.assertTrue(records[1].is_anomaly)
        self.assertEqual(records[1].mask_path.name, "1_label.mat")

    def test_list_image_files_sorts_numeric_names_and_ignores_non_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame_dir = Path(tmp)
            for name in ["010.jpg", "002.jpg", "Thumbs.db", "001.png"]:
                (frame_dir / name).write_bytes(b"x")

            frames = list_image_files(frame_dir)

        self.assertEqual([path.name for path in frames], ["001.png", "002.jpg", "010.jpg"])

    def test_load_frame_mask_reduces_pixel_like_masks_to_frame_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.npy"
            np.save(
                path,
                np.array(
                    [
                        [[0, 0], [0, 0]],
                        [[0, 1], [0, 0]],
                        [[0, 0], [2, 0]],
                    ],
                    dtype=np.uint8,
                ),
            )

            labels = load_frame_mask(path)

        self.assertEqual(labels.tolist(), [0, 1, 1])

    def test_load_frame_mask_reads_avenue_vol_label_mat(self):
        from scipy.io import savemat

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1_label.mat"
            cells = np.empty((3,), dtype=object)
            cells[0] = np.zeros((2, 2), dtype=np.uint8)
            cells[1] = np.array([[0, 1], [0, 0]], dtype=np.uint8)
            cells[2] = np.zeros((2, 2), dtype=np.uint8)
            savemat(path, {"volLabel": cells})

            labels = load_frame_mask(path)

        self.assertEqual(labels.tolist(), [0, 1, 0])

    def test_materialize_frame_dir_copies_to_one_based_processed_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            for name in ["002.jpg", "000.jpg", "001.jpg"]:
                (source / name).write_bytes(name.encode("utf-8"))
            config = PreprocessConfig(
                raw_root=root / "raw",
                processed_root=root / "processed",
                overwrite=True,
                num_workers=1,
            )
            record = VideoRecord(
                video_id="01_0014",
                split=TEST_SPLIT,
                class_name="scene_01",
                is_anomaly=False,
                source_path=source,
                frame_dir=config.frames_root / TEST_SPLIT / "01_0014",
                source_type=FRAME_DIR_SOURCE,
            )

            prepared = materialize_frame_dir(record, config)

            self.assertEqual(prepared.num_frames, 3)
            self.assertTrue((record.frame_dir / "000001.jpg").is_file())
            self.assertTrue((record.frame_dir / "000002.jpg").is_file())
            self.assertEqual(
                [frame_map.raw_frame_idx for frame_map in prepared.frame_maps],
                [0, 1, 2],
            )
            self.assertEqual((record.frame_dir / "000001.jpg").read_bytes(), b"000.jpg")

    def test_materialize_frame_dir_parallel_matches_single_worker_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            for idx in range(6):
                (source / f"{idx:03d}.jpg").write_bytes(f"frame-{idx}".encode("utf-8"))

            single_config = PreprocessConfig(
                raw_root=root / "raw",
                processed_root=root / "single",
                overwrite=True,
                num_workers=1,
            )
            parallel_config = PreprocessConfig(
                raw_root=root / "raw",
                processed_root=root / "parallel",
                overwrite=True,
                num_workers=4,
            )
            single_record = VideoRecord(
                video_id="01_0014",
                split=TEST_SPLIT,
                class_name="scene_01",
                is_anomaly=False,
                source_path=source,
                frame_dir=single_config.frames_root / TEST_SPLIT / "01_0014",
                source_type=FRAME_DIR_SOURCE,
            )
            parallel_record = VideoRecord(
                video_id="01_0014",
                split=TEST_SPLIT,
                class_name="scene_01",
                is_anomaly=False,
                source_path=source,
                frame_dir=parallel_config.frames_root / TEST_SPLIT / "01_0014",
                source_type=FRAME_DIR_SOURCE,
            )

            single = materialize_frame_dir(single_record, single_config)
            parallel = materialize_frame_dir(parallel_record, parallel_config)

            single_bytes = [
                (single_record.frame_dir / f"{idx:06d}.jpg").read_bytes() for idx in range(1, 7)
            ]
            parallel_bytes = [
                (parallel_record.frame_dir / f"{idx:06d}.jpg").read_bytes() for idx in range(1, 7)
            ]

        self.assertEqual(single.num_frames, parallel.num_frames)
        self.assertEqual(
            [frame_map.raw_frame_idx for frame_map in single.frame_maps],
            [frame_map.raw_frame_idx for frame_map in parallel.frame_maps],
        )
        self.assertEqual(single_bytes, parallel_bytes)

    def test_materialize_frame_files_converts_png_to_jpg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            target = root / "target.jpg"
            Image.new("RGB", (4, 4), color=(255, 0, 0)).save(source)
            config = PreprocessConfig(
                raw_root=root / "raw",
                processed_root=root / "processed",
                image_ext="jpg",
                image_backend="auto",
                num_workers=2,
            )

            materialize_frame_files([(source, target)], config)

            with Image.open(target) as converted:
                size = converted.size
                mode = converted.mode

        self.assertEqual(size, (4, 4))
        self.assertEqual(mode, "RGB")

    def test_materialize_frame_files_propagates_parallel_conversion_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "bad.png"
            target = root / "target.jpg"
            second_target = root / "second.jpg"
            source.write_bytes(b"not an image")
            config = PreprocessConfig(
                raw_root=root / "raw",
                processed_root=root / "processed",
                image_ext="jpg",
                image_backend="pillow",
                num_workers=2,
            )

            with self.assertRaises(Exception):
                materialize_frame_files([(source, target), (source, second_target)], config)

    def test_labels_for_record_maps_selected_source_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask_path = root / "mask.npy"
            np.save(mask_path, np.array([0, 1, 0, 1], dtype=np.uint8))
            record = VideoRecord(
                video_id="01_0014",
                split=TEST_SPLIT,
                class_name="scene_01",
                is_anomaly=True,
                source_path=root / "source",
                frame_dir=root / "frames",
                source_type=FRAME_DIR_SOURCE,
                mask_path=mask_path,
            )
            prepared = PreparedVideo(
                num_frames=2,
                effective_fps=None,
                frame_maps=[
                    FrameMap(1, root / "frames" / "000001.jpg", 1, None, None, 1),
                    FrameMap(2, root / "frames" / "000002.jpg", 3, None, None, 3),
                ],
            )

            labels = labels_for_record(record, prepared, PreprocessConfig(root, root))

        self.assertEqual(labels.tolist(), [1, 1])

    def test_context_future_windows_dense_with_gap_and_stride(self):
        sampling = SamplingConfig(
            context_frames=3,
            future_frames=2,
            context_sampling="dense",
            context_span=3,
            gap=1,
            stride_train=2,
            stride_test=2,
        )

        windows = list(
            context_future_windows(
                10,
                sampling,
                stride=2,
                frame_index_start=1,
            )
        )

        self.assertEqual(
            windows,
            [
                ([1, 2, 3], [5, 6]),
                ([3, 4, 5], [7, 8]),
                ([5, 6, 7], [9, 10]),
            ],
        )

    def test_build_sample_indices_writes_train_and_test_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed"
            train_frames = processed / "frames" / TRAIN_SPLIT / "01_001"
            test_frames = processed / "frames" / TEST_SPLIT / "01_0014"
            train_frames.mkdir(parents=True)
            test_frames.mkdir(parents=True)
            config = PreprocessConfig(
                raw_root=root / "raw",
                processed_root=processed,
                sampling=SamplingConfig(
                    context_frames=2,
                    future_frames=2,
                    context_sampling="dense",
                    context_span=2,
                    gap=0,
                    stride_train=2,
                    stride_test=1,
                ),
            )
            with config.manifest_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
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
                writer.writeheader()
                writer.writerow(
                    {
                        "video_id": "01_001",
                        "split": TRAIN_SPLIT,
                        "class_name": "scene_01",
                        "is_anomaly": 0,
                        "source_path": "raw.avi",
                        "frame_dir": str(train_frames),
                        "num_frames": 6,
                        "effective_fps": "",
                    }
                )
                writer.writerow(
                    {
                        "video_id": "01_0014",
                        "split": TEST_SPLIT,
                        "class_name": "scene_01",
                        "is_anomaly": 1,
                        "source_path": "raw_frames",
                        "frame_dir": str(test_frames),
                        "num_frames": 5,
                        "effective_fps": "",
                    }
                )
            with config.frame_labels_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "video_id",
                        "split",
                        "class_name",
                        "frame_idx",
                        "frame_path",
                        "label",
                    ],
                )
                writer.writeheader()
                for idx in range(1, 7):
                    writer.writerow(
                        {
                            "video_id": "01_001",
                            "split": TRAIN_SPLIT,
                            "class_name": "scene_01",
                            "frame_idx": idx,
                            "frame_path": str(train_frames / f"{idx:06d}.jpg"),
                            "label": 0,
                        }
                    )
                for idx in range(1, 6):
                    writer.writerow(
                        {
                            "video_id": "01_0014",
                            "split": TEST_SPLIT,
                            "class_name": "scene_01",
                            "frame_idx": idx,
                            "frame_path": str(test_frames / f"{idx:06d}.jpg"),
                            "label": int(idx == 4),
                        }
                    )

            train_count, test_count = build_sample_indices(config)
            train_records = read_jsonl(config.train_samples_path)
            test_records = read_jsonl(config.test_samples_path)

        self.assertEqual(train_count, 2)
        self.assertEqual(test_count, 2)
        self.assertEqual(train_records[0]["future_frames"], [3, 4])
        self.assertEqual(test_records[0]["future_label"], 1)
        self.assertEqual(test_records[1]["future_label"], 1)
        self.assertTrue(test_records[0]["future_frame_paths"][0].endswith("000003.jpg"))


if __name__ == "__main__":
    unittest.main()
