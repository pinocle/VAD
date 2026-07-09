import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
import yaml
from src.pipelines.feature_eng_pipeline import read_jsonl, write_jsonl
from src.pipelines.inference_pipeline import infer
from src.pipelines.object_track_pipeline import (
    TRACK_CHANNELS,
    TrackObject,
    build_track_feature_index,
    load_raw_track_cache,
    load_track_feature_payload,
    parse_track_frame,
    rasterize_frame_grid,
    rasterize_sample_grids,
    write_track_cache_jsonl,
)
from src.pipelines.training_pipeline import train
from src.pipelines.z_dit_pipeline import (
    load_pipeline_config,
    per_frame_z_score_variants,
)


class ObjectTrackPipelineTests(unittest.TestCase):
    def test_raw_cache_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracks.jsonl"
            write_track_cache_jsonl(
                path,
                [
                    {
                        "video_id": "video",
                        "frame_idx": 1,
                        "image_size": [256, 256],
                        "objects": [
                            {
                                "track_id": "t1",
                                "bbox_xyxy_norm": [0.1, 0.2, 0.3, 0.4],
                                "score": 0.9,
                                "velocity_xy_norm": [0.01, -0.02],
                            }
                        ],
                    }
                ],
            )

            cache = load_raw_track_cache(path)

        self.assertIn(("video", 1), cache)
        self.assertEqual(cache[("video", 1)][0].track_id, "t1")
        self.assertEqual(cache[("video", 1)][0].velocity, (0.01, -0.02))

    def test_malformed_bbox_validation(self):
        with self.assertRaisesRegex(ValueError, "bbox_xyxy_norm"):
            parse_track_frame(
                {
                    "video_id": "video",
                    "frame_idx": 1,
                    "objects": [
                        {
                            "track_id": "t1",
                            "bbox_xyxy_norm": [0.5, 0.1, 0.4, 0.2],
                            "score": 0.9,
                        }
                    ],
                }
            )

    def test_missing_velocity_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracks.jsonl"
            write_track_cache_jsonl(
                path,
                [
                    {
                        "video_id": "video",
                        "frame_idx": 1,
                        "objects": [
                            {
                                "track_id": "t1",
                                "bbox_xyxy_norm": [0.1, 0.1, 0.3, 0.3],
                                "score": 1.0,
                            }
                        ],
                    },
                    {
                        "video_id": "video",
                        "frame_idx": 2,
                        "objects": [
                            {
                                "track_id": "t1",
                                "bbox_xyxy_norm": [0.3, 0.1, 0.5, 0.3],
                                "score": 1.0,
                            }
                        ],
                    },
                ],
            )

            cache = load_raw_track_cache(path)

        self.assertAlmostEqual(cache[("video", 2)][0].velocity[0], 0.2)
        self.assertAlmostEqual(cache[("video", 2)][0].velocity[1], 0.0)

    def test_empty_object_frame_rasterizes_to_zero(self):
        sample = {
            "sample_id": "sample",
            "video_id": "video",
            "context_frames": [1],
            "future_frames": [2],
        }

        payload = rasterize_sample_grids(sample, {}, grid_size=(2, 2))

        self.assertEqual(tuple(payload["context_grid"].shape), (1, len(TRACK_CHANNELS), 2, 2))
        self.assertEqual(float(payload["context_grid"].sum()), 0.0)
        self.assertEqual(float(payload["future_grid"].sum()), 0.0)

    def test_single_box_grid_rasterization(self):
        obj = TrackObject(
            video_id="video",
            frame_idx=1,
            track_id="t1",
            bbox_xyxy_norm=(0.25, 0.25, 0.75, 0.75),
            score=0.5,
            velocity_xy_norm=(0.1, 0.0),
        )

        grid = rasterize_frame_grid([obj], grid_size=(4, 4))

        self.assertTrue(torch.equal(grid[0, 1:3, 1:3], torch.full((2, 2), 0.5)))
        self.assertAlmostEqual(float(grid[0].sum()), 2.0)
        self.assertTrue(torch.equal(grid[1, 1:3, 1:3], torch.full((2, 2), 0.1)))

    def test_trajectory_rasterization_is_causal(self):
        sample = {
            "sample_id": "sample",
            "video_id": "video",
            "context_frames": [1, 2],
            "future_frames": [3],
        }
        objects = {
            ("video", 1): (make_track_object(frame_idx=1, bbox=(0.00, 0.00, 0.20, 0.20)),),
            ("video", 2): (make_track_object(frame_idx=2, bbox=(0.25, 0.00, 0.45, 0.20)),),
            ("video", 3): (make_track_object(frame_idx=3, bbox=(0.50, 0.00, 0.70, 0.20)),),
        }

        payload = rasterize_sample_grids(sample, objects, grid_size=(4, 4))

        self.assertAlmostEqual(float(payload["context_grid"][0, 4].sum()), 1.0)
        self.assertAlmostEqual(float(payload["future_grid"][0, 4].sum()), 3.0)

    def test_build_track_feature_index_writes_augmented_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_index = make_feature_project(root)
            track_cache = root / "tracks.jsonl"
            write_track_cache_jsonl(
                track_cache,
                [
                    {
                        "video_id": "video",
                        "frame_idx": 4,
                        "objects": [
                            {
                                "track_id": "t1",
                                "bbox_xyxy_norm": [0.0, 0.0, 0.5, 0.5],
                                "score": 1.0,
                            }
                        ],
                    }
                ],
            )
            output_index = root / "features" / "test_feature_index_tracks.jsonl"

            records = build_track_feature_index(
                feature_index_path=feature_index,
                track_cache_path=track_cache,
                output_feature_index_path=output_index,
                z_patch_size=2,
            )
            payload = load_track_feature_payload(Path(records[0]["track_feature_path"]))

            self.assertTrue(output_index.is_file())
            self.assertIn("track_feature_path", read_jsonl(output_index)[0])
            self.assertEqual(tuple(payload["future_grid"].shape[-2:]), (2, 2))

    def test_grid_patch_shape_mismatch_error(self):
        scoring = track_scoring_config(("track_weighted",), "track_weighted")
        z = torch.ones(1, 1, 1, 4, 4)
        z_hat = torch.zeros_like(z)
        stats = {"mean": torch.zeros(1, 1, 1, 1, 1), "std": torch.ones(1, 1, 1, 1, 1)}
        track_grid = torch.zeros(1, 1, len(TRACK_CHANNELS), 3, 3)

        with self.assertRaisesRegex(ValueError, "H_grid\\*W_grid"):
            per_frame_z_score_variants(
                z,
                z_hat,
                scoring=scoring,
                patch_size=2,
                z_stats=stats,
                track_grid=track_grid,
            )

    def test_track_weighted_equals_mean_when_grid_empty(self):
        scoring = track_scoring_config(("global", "track_weighted"), "track_weighted")
        z = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 1, 4, 4)
        z_hat = torch.zeros_like(z)
        stats = {"mean": torch.zeros(1, 1, 1, 1, 1), "std": torch.ones(1, 1, 1, 1, 1)}
        track_grid = torch.zeros(1, 1, len(TRACK_CHANNELS), 2, 2)

        variants = per_frame_z_score_variants(
            z,
            z_hat,
            scoring=scoring,
            patch_size=2,
            z_stats=stats,
            track_grid=track_grid,
        )

        self.assertAlmostEqual(
            float(variants["track_weighted"][0, 0]), float(variants["global"][0, 0])
        )

    def test_track_region_topk_increases_when_high_error_overlaps_object(self):
        scoring = track_scoring_config(("global", "track_region_topk"), "track_region_topk")
        z = torch.ones(1, 1, 1, 4, 4)
        z[:, :, :, :2, :2] = 10.0
        z_hat = torch.zeros_like(z)
        stats = {"mean": torch.zeros(1, 1, 1, 1, 1), "std": torch.ones(1, 1, 1, 1, 1)}
        track_grid = torch.zeros(1, 1, len(TRACK_CHANNELS), 2, 2)
        track_grid[:, :, 0, 0, 0] = 1.0

        variants = per_frame_z_score_variants(
            z,
            z_hat,
            scoring=scoring,
            patch_size=2,
            z_stats=stats,
            track_grid=track_grid,
        )

        self.assertGreater(
            float(variants["track_region_topk"][0, 0]), float(variants["global"][0, 0])
        )

    def test_inference_smoke_writes_track_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, checkpoint_root, feature_index = make_track_smoke_project(root)
            track_cache = root / "tracks.jsonl"
            write_track_cache_jsonl(
                track_cache,
                [
                    {
                        "video_id": "video",
                        "frame_idx": 4,
                        "objects": [
                            {
                                "track_id": "t1",
                                "bbox_xyxy_norm": [0.0, 0.0, 0.5, 0.5],
                                "score": 1.0,
                            }
                        ],
                    },
                    {
                        "video_id": "video",
                        "frame_idx": 5,
                        "objects": [
                            {
                                "track_id": "t1",
                                "bbox_xyxy_norm": [0.5, 0.5, 1.0, 1.0],
                                "score": 1.0,
                            }
                        ],
                    },
                ],
            )
            augmented_index = root / "features" / "test_feature_index_tracks.jsonl"
            build_track_feature_index(
                feature_index_path=feature_index,
                track_cache_path=track_cache,
                output_feature_index_path=augmented_index,
                z_patch_size=2,
            )

            train(config_path, run_id="smoke", max_steps=1, limit_samples=2)
            output_dir = infer(
                config_path,
                checkpoint_path=checkpoint_root / "smoke" / "best.pt",
                run_id="smoke",
                feature_index=augmented_index,
                limit_samples=2,
                overwrite=True,
            )
            frame_scores = read_jsonl(output_dir / "frame_scores.jsonl")
            predictions = read_jsonl(output_dir / "future_frame_predictions.jsonl")
            metrics = yaml.safe_load((output_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertIn("global_score", frame_scores[0])
        self.assertIn("track_weighted_score", frame_scores[0])
        self.assertIn("track_region_topk_score", frame_scores[0])
        self.assertIn("score", frame_scores[0])
        self.assertIn("track_weighted", predictions[0]["future_frame_score_variants"])
        self.assertIn("track_weighted_score", metrics["score_metrics"])


def make_track_object(
    *,
    frame_idx: int,
    bbox: tuple[float, float, float, float],
) -> TrackObject:
    return TrackObject(
        video_id="video",
        frame_idx=frame_idx,
        track_id="t1",
        bbox_xyxy_norm=bbox,
        score=1.0,
        velocity_xy_norm=(0.0, 0.0),
    )


def track_scoring_config(variants: tuple[str, ...], primary: str):
    config = load_pipeline_config(Path("config/local.yaml"))
    return replace(
        config.scoring,
        variants=variants,
        variant=primary,
        score_normalization=replace(config.scoring.score_normalization, enabled=False),
    )


def make_feature_project(root: Path) -> Path:
    features = root / "features"
    records = make_feature_records(features, "test", count=1, labels=[0])
    feature_index = features / "test_feature_index.jsonl"
    write_jsonl(feature_index, records)
    return feature_index


def make_track_smoke_project(root: Path) -> tuple[Path, Path, Path]:
    features = root / "features"
    checkpoints = root / "checkpoints"
    predictions = root / "predictions"
    train_records = make_feature_records(features, "train", count=3, labels=[0, 0, 0])
    test_records = make_feature_records(features, "test", count=2, labels=[0, 1])
    train_index = features / "train_feature_index.jsonl"
    test_index = features / "test_feature_index.jsonl"
    write_jsonl(train_index, train_records)
    write_jsonl(test_index, test_records)
    config = {
        "dataset": {"name": "shanghaitech"},
        "model": {
            "name": "high_low_conditioned_z_dit",
            "prediction_type": "velocity",
            "dit": {
                "hidden_size": 16,
                "num_layers": 1,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
            },
            "high_adapter": {
                "input_dim": "auto",
                "max_tokens": 8,
                "token_reduction": "uniform",
                "use_pos_embedding": True,
            },
            "z_adapter": {
                "patch_size": 2,
                "use_temporal_pos": True,
                "use_spatial_pos": True,
            },
            "low_adapter": {
                "patch_size": 8,
                "use_temporal_pos": True,
                "use_spatial_pos": True,
            },
        },
        "flow_matching": {
            "inference_steps": 2,
            "timestep_distribution": "uniform",
            "time_embedding_scale": 1000.0,
            "normalize_z": True,
            "beta_alpha": 1.5,
            "beta_beta": 1.0,
            "beta_s": 0.999,
        },
        "loss": {"alpha": 0.2, "topk_fraction": 0.1},
        "optimization": {"matmul_precision": "high"},
        "training": {
            "feature_index": str(train_index),
            "output_root": str(checkpoints),
            "run_id": None,
            "batch_size": 2,
            "num_workers": 0,
            "max_steps": 1,
            "val_fraction": 0.0,
            "learning_rate": 1.0e-4,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "dtype": "float32",
            "amp": False,
            "seed": 7,
            "log_every_steps": 1,
            "save_every_steps": 1,
        },
        "inference": {
            "feature_index": str(test_index),
            "checkpoint_path": None,
            "output_root": str(predictions),
            "batch_size": 2,
            "num_workers": 0,
            "mode": "offline_cached",
            "save_tensors": False,
            "overwrite": True,
        },
        "scoring": {
            "method": "normalized_z_mse",
            "frame_aggregation": "mean",
            "variant": "track_weighted",
            "variants": [
                "global",
                "track_weighted",
                "track_region_topk",
                "mean_topk_plus_track_region",
            ],
            "topk_fraction": 0.25,
            "beta": 0.2,
            "track_region_beta": 0.2,
            "track_weight_max": 5.0,
            "score_normalization": {
                "enabled": False,
                "primary": "raw",
                "rolling_window": 128,
                "rolling_min_history": 16,
                "rolling_windows": [],
            },
            "normalized_z_mse": {"eps": 1.0e-6},
            "decoded_mse": {"enabled": False, "alpha": 0.1},
        },
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, checkpoints, test_index


def make_feature_records(
    features: Path,
    split: str,
    *,
    count: int,
    labels: list[int],
) -> list[dict[str, object]]:
    records = []
    for index in range(count):
        sample_id = f"{split}_{index:03d}"
        high_path = features / "high" / split / f"{sample_id}.pt"
        z_path = features / "z" / split / f"{sample_id}.pt"
        low_path = features / "low" / split / f"{sample_id}.pt"
        high_path.parent.mkdir(parents=True, exist_ok=True)
        z_path.parent.mkdir(parents=True, exist_ok=True)
        low_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"sample_id": sample_id, "high": torch.randn(3, 5, 6)}, high_path)
        torch.save({"sample_id": sample_id, "z": torch.randn(2, 2, 4, 4)}, z_path)
        torch.save({"sample_id": sample_id, "low": torch.randn(2, 3, 16, 16)}, low_path)
        records.append(
            {
                "sample_id": sample_id,
                "video_id": "video",
                "split": split,
                "scene_id": "scene",
                "context_frames": [index + 1, index + 2, index + 3],
                "future_frames": [index + 4, index + 5],
                "future_frame_labels": [0, int(labels[index])],
                "future_label": int(labels[index]),
                "high_feature_path": str(high_path),
                "z_path": str(z_path),
                "low_feature_path": str(low_path),
            }
        )
    return records


if __name__ == "__main__":
    unittest.main()
