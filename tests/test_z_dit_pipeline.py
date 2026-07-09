import json
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from src.models.z_dit import (
    CONDITION_MODE_APPEARANCE_ONLY,
    CONDITION_MODE_APPEARANCE_TRACK,
    CONDITION_MODE_TRACK_ONLY,
    ConditionedZDiT,
    FlowMatchingSampler,
    HighAdapter,
    PatchConditionAdapter,
    TrackGridAdapter,
    ZDiTShape,
    ZPatchEmbed,
)
from src.pipelines.feature_eng_pipeline import read_jsonl, write_jsonl
from src.pipelines.inference_pipeline import infer
from src.pipelines.object_track_pipeline import TRACK_CHANNELS
from src.pipelines.training_pipeline import compute_velocity_loss, topk_patch_mse, train
from src.pipelines.z_dit_pipeline import (
    FeatureZDataset,
    aggregate_frame_scores,
    apply_score_calibration,
    apply_score_normalization,
    compute_z_stats,
    fit_score_calibration,
    load_feature_records,
    load_pipeline_config,
    per_frame_z_score_variants,
    per_frame_z_scores,
)


class ZDiTPipelineTests(unittest.TestCase):
    def test_local_config_loads_z_dit_sections(self):
        config = load_pipeline_config(Path("config/local.yaml"))

        self.assertEqual(config.model.name, "high_low_conditioned_z_dit")
        self.assertEqual(config.model.prediction_type, "velocity")
        self.assertTrue(config.model.dit.gradient_checkpointing)
        self.assertEqual(config.optimization.matmul_precision, "high")
        self.assertTrue(config.optimization.enable_tf32)
        self.assertTrue(config.optimization.cudnn_benchmark)
        self.assertTrue(config.optimization.enable_flash_sdp)
        self.assertEqual(config.optimization.fused_adamw, "auto")
        self.assertEqual(config.optimization.prefetch_factor, 1)
        self.assertIsNone(config.training.init_checkpoint_path)
        self.assertFalse(config.training.overwrite)
        self.assertFalse(config.training.compile)
        self.assertFalse(config.inference.compile)
        self.assertEqual(config.flow_matching.timestep_distribution, "gr00t_beta")
        self.assertEqual(config.flow_matching.inference_steps, 4)
        self.assertEqual(config.model.high_adapter.token_reduction, "aligned_grid")
        self.assertEqual(config.model.high_adapter.foreground_ratio, 0.75)
        self.assertEqual(config.model.z_adapter.patch_size, 2)
        self.assertEqual(config.model.low_adapter.patch_size, 16)
        self.assertEqual(config.model.condition.mode, "baseline")
        self.assertFalse(config.model.condition.use_track_grid)
        self.assertEqual(config.model.condition.track_grid_gate, 0.1)
        self.assertEqual(config.model.condition.track_grid.ablation_mode, "real")
        self.assertEqual(config.model.condition.track_grid.channel_mask, "all_channels")
        self.assertIsNone(config.model.condition.track_grid.gate_override)
        self.assertEqual(config.loss.alpha, 0.2)
        self.assertEqual(config.loss.topk_fraction, 0.1)
        self.assertEqual(config.scoring.method, "normalized_z_mse")
        self.assertEqual(config.scoring.variant, "global_patch")
        self.assertEqual(config.scoring.beta, 0.2)
        self.assertEqual(config.scoring.motion_topk_fraction, 0.25)
        self.assertIn(0.5, config.scoring.sweep.betas)
        self.assertIn(0.2, config.scoring.sweep.topk_fractions)
        self.assertIn("global", config.scoring.variants)
        self.assertIn("patch", config.scoring.variants)
        self.assertIn("low_weighted", config.scoring.variants)
        self.assertIn("global_patch", config.scoring.variants)
        self.assertIn("motion_patch", config.scoring.variants)
        self.assertIn("motion_global_patch", config.scoring.variants)
        self.assertTrue(config.scoring.score_normalization.enabled)
        self.assertEqual(config.scoring.score_normalization.primary, "video_centered")
        self.assertIn(64, config.scoring.score_normalization.rolling_windows)

    def test_z_patchify_unpatchify_roundtrip(self):
        adapter = ZPatchEmbed(
            future_frames=2,
            z_channels=3,
            z_height=4,
            z_width=4,
            patch_size=2,
            hidden_size=16,
            use_temporal_pos=True,
            use_spatial_pos=True,
        )
        z = torch.arange(2 * 2 * 3 * 4 * 4, dtype=torch.float32).view(2, 2, 3, 4, 4)

        patches = adapter.patchify(z)
        restored = adapter.unpatchify(patches)

        self.assertEqual(tuple(patches.shape), (2, 8, 12))
        self.assertTrue(torch.equal(restored, z))

    def test_high_adapter_aligns_patch_grid_to_z_grid(self):
        adapter = HighAdapter(
            input_dim=3,
            high_frames=2,
            high_tokens=16,
            hidden_size=8,
            max_tokens=4,
            use_pos_embedding=False,
            token_reduction="aligned_grid",
            aligned_frames=3,
            aligned_grid_h=2,
            aligned_grid_w=2,
        )
        high = torch.randn(2, 2, 16, 3)

        high_tokens = adapter(high)
        future_tokens = adapter.future_aligned_tokens(high_tokens)

        self.assertEqual(tuple(high_tokens.shape), (2, 8, 8))
        self.assertEqual(tuple(future_tokens.shape), (2, 12, 8))

    def test_track_grid_adapter_shape(self):
        adapter = TrackGridAdapter(
            track_frames=3,
            track_channels=len(TRACK_CHANNELS),
            grid_h=2,
            grid_w=3,
            hidden_size=8,
        )
        grid = torch.randn(2, 3, len(TRACK_CHANNELS), 2, 3)

        tokens = adapter(grid)

        self.assertEqual(tuple(tokens.shape), (2, 18, 8))

    def test_patch_condition_adapter_shape(self):
        adapter = PatchConditionAdapter(
            condition_frames=3,
            input_channels=8,
            grid_h=2,
            grid_w=3,
            hidden_size=16,
        )
        grid = torch.randn(2, 3, 8, 2, 3)

        tokens = adapter(grid)

        self.assertEqual(tuple(tokens.shape), (2, 18, 16))

    def test_model_forward_appearance_only_patch_condition(self):
        prediction, model, z = run_patch_condition_forward(CONDITION_MODE_APPEARANCE_ONLY)

        self.assertEqual(tuple(prediction.shape), tuple(z.shape))
        self.assertIsNone(model.high_adapter)
        self.assertIsNone(model.track_grid_adapter)
        self.assertIsNone(model.low_adapter)
        self.assertIsNotNone(model.patch_condition_adapter)

    def test_model_forward_track_only_patch_condition(self):
        prediction, model, z = run_patch_condition_forward(CONDITION_MODE_TRACK_ONLY)

        self.assertEqual(tuple(prediction.shape), tuple(z.shape))
        self.assertIsNone(model.high_adapter)
        self.assertIsNone(model.track_grid_adapter)
        self.assertFalse(hasattr(model, "track_grid_gate"))
        self.assertIsNotNone(model.patch_condition_adapter)

    def test_model_forward_appearance_track_patch_condition(self):
        prediction, model, z = run_patch_condition_forward(CONDITION_MODE_APPEARANCE_TRACK)

        self.assertEqual(tuple(prediction.shape), tuple(z.shape))
        self.assertEqual(model.patch_condition_adapter.input_channels, 3 + len(TRACK_CHANNELS))
        self.assertEqual(model.patch_condition_adapter.condition_frames, 3)

    def test_model_forward_and_flow_matching_shapes(self):
        shape = ZDiTShape(
            high_frames=3,
            high_tokens=5,
            high_dim=6,
            future_frames=2,
            z_channels=2,
            z_height=4,
            z_width=4,
            low_frames=2,
            low_channels=3,
            low_height=8,
            low_width=8,
        )
        model = ConditionedZDiT(
            shape=shape,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            high_max_tokens=8,
            high_use_pos_embedding=True,
            high_token_reduction="foreground_background",
            high_foreground_ratio=0.5,
            z_patch_size=2,
            z_use_temporal_pos=True,
            z_use_spatial_pos=True,
            low_patch_size=4,
            low_use_temporal_pos=True,
            low_use_spatial_pos=True,
        )
        self.assertIsNone(model.track_grid_adapter)
        flow = FlowMatchingSampler(
            inference_steps=2,
            timestep_distribution="uniform",
            time_embedding_scale=1000.0,
            normalize_z=True,
        )
        high = torch.randn(2, 3, 5, 6)
        low = torch.randn(2, 2, 3, 8, 8)
        z = torch.randn(2, 2, 2, 4, 4)
        z_stats = {
            "mean": torch.zeros(1, 1, 2, 1, 1),
            "std": torch.ones(1, 1, 2, 1, 1),
        }

        noised, time_values, target, clean = flow.prepare_training_pair(
            z,
            z_stats=z_stats,
        )
        prediction = model(noised, time_values, high, low)
        sampled = flow.euler_sample(
            model,
            high,
            tuple(z.shape),
            low_features=low,
            z_stats=z_stats,
            inference_steps=2,
        )

        self.assertEqual(tuple(noised.shape), tuple(z.shape))
        self.assertEqual(tuple(target.shape), tuple(z.shape))
        self.assertEqual(tuple(clean.shape), tuple(z.shape))
        self.assertEqual(tuple(prediction.shape), tuple(z.shape))
        self.assertEqual(tuple(sampled.shape), tuple(z.shape))

    def test_model_forward_with_track_grid_condition(self):
        shape = ZDiTShape(
            high_frames=3,
            high_tokens=5,
            high_dim=6,
            future_frames=2,
            z_channels=2,
            z_height=4,
            z_width=4,
            low_frames=2,
            low_channels=3,
            low_height=8,
            low_width=8,
            track_frames=3,
            track_channels=len(TRACK_CHANNELS),
            track_grid_h=2,
            track_grid_w=2,
        )
        model = ConditionedZDiT(
            shape=shape,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            high_max_tokens=8,
            high_use_pos_embedding=True,
            high_token_reduction="uniform",
            high_foreground_ratio=0.5,
            z_patch_size=2,
            z_use_temporal_pos=True,
            z_use_spatial_pos=True,
            low_patch_size=4,
            low_use_temporal_pos=True,
            low_use_spatial_pos=True,
            use_track_grid=True,
            track_grid_gate_init=0.1,
        )
        high = torch.randn(2, 3, 5, 6)
        low = torch.randn(2, 2, 3, 8, 8)
        track_grid = torch.randn(2, 3, len(TRACK_CHANNELS), 2, 2)
        z = torch.randn(2, 2, 2, 4, 4)
        time_values = torch.rand(2)

        prediction = model(z, time_values, high, low, track_grid)

        self.assertEqual(tuple(prediction.shape), tuple(z.shape))
        self.assertAlmostEqual(float(model.track_grid_gate.detach()), 0.1)

    def test_model_forward_with_aligned_high_grid(self):
        shape = ZDiTShape(
            high_frames=3,
            high_tokens=16,
            high_dim=6,
            future_frames=2,
            z_channels=2,
            z_height=4,
            z_width=4,
            low_frames=2,
            low_channels=3,
            low_height=8,
            low_width=8,
        )
        model = ConditionedZDiT(
            shape=shape,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            high_max_tokens=8,
            high_use_pos_embedding=True,
            high_token_reduction="aligned_grid",
            high_foreground_ratio=0.5,
            z_patch_size=2,
            z_use_temporal_pos=True,
            z_use_spatial_pos=True,
            low_patch_size=4,
            low_use_temporal_pos=True,
            low_use_spatial_pos=True,
        )
        high = torch.randn(2, 3, 16, 6)
        low = torch.randn(2, 2, 3, 8, 8)
        z = torch.randn(2, 2, 2, 4, 4)
        time_values = torch.rand(2)

        prediction = model(z, time_values, high, low)

        self.assertEqual(tuple(prediction.shape), tuple(z.shape))

    def test_model_rejects_misaligned_low_grid(self):
        shape = ZDiTShape(
            high_frames=3,
            high_tokens=16,
            high_dim=6,
            future_frames=2,
            z_channels=2,
            z_height=4,
            z_width=4,
            low_frames=2,
            low_channels=3,
            low_height=8,
            low_width=8,
        )

        with self.assertRaisesRegex(ValueError, "C_low patch grid must match Z patch grid"):
            ConditionedZDiT(
                shape=shape,
                hidden_size=16,
                num_layers=1,
                num_heads=4,
                mlp_ratio=2.0,
                dropout=0.0,
                high_max_tokens=8,
                high_use_pos_embedding=True,
                high_token_reduction="aligned_grid",
                high_foreground_ratio=0.5,
                z_patch_size=2,
                z_use_temporal_pos=True,
                z_use_spatial_pos=True,
                low_patch_size=8,
                low_use_temporal_pos=True,
                low_use_spatial_pos=True,
            )

    def test_scoring_and_frame_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, _ = make_fake_project(root)
            config = load_pipeline_config(config_path)
            records = read_jsonl(root / "features" / "train_feature_index.jsonl")
            dataset = FeatureZDataset(records)
            stats = compute_z_stats(dataset)
            sample = dataset[0]
            z = sample["z"].unsqueeze(0)
            z_hat = torch.zeros_like(z)

            scores = per_frame_z_scores(
                z,
                z_hat,
                scoring=config.scoring,
                patch_size=config.model.z_adapter.patch_size,
                z_stats=stats,
            )
            prediction_records = [
                {
                    "video_id": "video",
                    "future_frames": [1, 2],
                    "future_frame_labels": [0, 1],
                    "future_frame_scores": [float(scores[0, 0]), float(scores[0, 1])],
                },
                {
                    "video_id": "video",
                    "future_frames": [2, 3],
                    "future_frame_labels": [1, 0],
                    "future_frame_scores": [1.0, 3.0],
                },
            ]

            frame_scores = apply_score_normalization(
                aggregate_frame_scores(prediction_records),
                scoring=config.scoring,
            )

        self.assertEqual([record["frame_idx"] for record in frame_scores], [1, 2, 3])
        self.assertEqual(frame_scores[1]["num_votes"], 2)
        self.assertIn("raw_score", frame_scores[0])
        self.assertIn("video_centered_score", frame_scores[0])
        self.assertIn("rolling_centered_score", frame_scores[0])
        self.assertIn("global_raw_score", frame_scores[0])
        self.assertIn("global_rolling_w64_centered_score", frame_scores[0])

    def test_missing_track_feature_path_raises_when_track_condition_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_fake_project(root)

            with self.assertRaisesRegex(ValueError, "track_feature_path"):
                load_feature_records(
                    root / "features" / "train_feature_index.jsonl",
                    normal_only=True,
                    limit_samples=None,
                    require_track_features=True,
                )

    def test_z_score_variants_use_patch_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, _ = make_fake_project(root)
            config = load_pipeline_config(config_path)
            z = torch.tensor(
                [
                    [
                        [
                            [
                                [1.0, 1.0, 2.0, 2.0],
                                [1.0, 1.0, 2.0, 2.0],
                                [3.0, 3.0, 4.0, 4.0],
                                [3.0, 3.0, 4.0, 4.0],
                            ]
                        ]
                    ]
                ]
            )
            z_hat = torch.zeros_like(z)
            stats = {
                "mean": torch.zeros(1, 1, 1, 1, 1),
                "std": torch.ones(1, 1, 1, 1, 1),
            }
            low = torch.zeros(1, 1, 3, 8, 8)
            low[:, :, :, 4:, 4:] = 1.0

            variants = per_frame_z_score_variants(
                z,
                z_hat,
                scoring=config.scoring,
                patch_size=config.model.z_adapter.patch_size,
                z_stats=stats,
                low=low,
            )

        self.assertAlmostEqual(float(variants["global"][0, 0]), 7.5)
        self.assertAlmostEqual(float(variants["patch"][0, 0]), 16.0)
        self.assertAlmostEqual(float(variants["low_weighted"][0, 0]), 11.75)
        self.assertAlmostEqual(float(variants["global_patch"][0, 0]), 10.7, places=6)
        self.assertAlmostEqual(float(variants["motion_patch"][0, 0]), 16.0)
        self.assertAlmostEqual(float(variants["motion_global_patch"][0, 0]), 10.7, places=6)
        self.assertIn("global_patch_b0p5_k0p2", variants)

    def test_scene_calibration_adds_group_normalized_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, _ = make_fake_project(root)
            config = load_pipeline_config(config_path)
            frame_scores = [
                {
                    "video_id": "a",
                    "scene_id": "scene_01",
                    "frame_idx": 1,
                    "label": 0,
                    "global_patch_raw_score": 1.0,
                    "score": 1.0,
                },
                {
                    "video_id": "a",
                    "scene_id": "scene_01",
                    "frame_idx": 2,
                    "label": 0,
                    "global_patch_raw_score": 3.0,
                    "score": 3.0,
                },
                {
                    "video_id": "a",
                    "scene_id": "scene_01",
                    "frame_idx": 3,
                    "label": 1,
                    "global_patch_raw_score": 5.0,
                    "score": 5.0,
                },
            ]

            stats = fit_score_calibration(frame_scores, scoring=config.scoring)
            calibrated = apply_score_calibration(
                frame_scores,
                scoring=config.scoring,
                calibration_stats=stats,
            )

        self.assertIn("global_patch_scene_calibrated_score", calibrated[-1])
        self.assertGreater(calibrated[-1]["global_patch_scene_calibrated_score"], 0.0)

    def test_velocity_loss_adds_topk_patch_term(self):
        prediction = torch.zeros(1, 1, 1, 4, 4)
        target = torch.tensor(
            [
                [
                    [
                        [
                            [1.0, 1.0, 2.0, 2.0],
                            [1.0, 1.0, 2.0, 2.0],
                            [3.0, 3.0, 4.0, 4.0],
                            [3.0, 3.0, 4.0, 4.0],
                        ]
                    ]
                ]
            ]
        )

        loss, loss_global, loss_patch = compute_velocity_loss(
            prediction,
            target,
            patch_size=2,
            patch_alpha=0.2,
            topk_fraction=0.25,
        )

        self.assertAlmostEqual(float(loss_global), 7.5)
        self.assertAlmostEqual(float(loss_patch), 16.0)
        self.assertAlmostEqual(float(loss), 10.7, places=6)
        self.assertAlmostEqual(
            float(topk_patch_mse(prediction, target, patch_size=2, topk_fraction=1.0)),
            7.5,
        )

    def test_train_and_infer_smoke_with_fake_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, checkpoint_root = make_fake_project(root)

            run_dir = train(config_path, run_id="smoke", max_steps=1, limit_samples=2)
            output_dir = infer(
                config_path,
                checkpoint_path=checkpoint_root / "smoke" / "best.pt",
                run_id="smoke",
                limit_samples=2,
                overwrite=True,
            )
            frame_scores = read_jsonl(output_dir / "frame_scores.jsonl")
            predictions = read_jsonl(output_dir / "future_frame_predictions.jsonl")
            metrics = read_jsonl(run_dir / "metrics.jsonl")
            checkpoint_exists = (run_dir / "best.pt").is_file()
            metrics_exists = (output_dir / "metrics.json").is_file()

        self.assertTrue(checkpoint_exists)
        self.assertTrue(metrics_exists)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(frame_scores)
        self.assertIn("score", frame_scores[0])
        self.assertIn("raw_score", frame_scores[0])
        self.assertIn("video_centered_score", frame_scores[0])
        self.assertIn("global_raw_score", frame_scores[0])
        self.assertIn("patch_raw_score", frame_scores[0])
        self.assertIn("global_patch_raw_score", frame_scores[0])
        self.assertIn("train_loss_global", metrics[0])
        self.assertIn("train_loss_patch", metrics[0])
        self.assertIn("train_loss_patch_weighted", metrics[0])

    def test_train_and_infer_smoke_with_appearance_track_patch_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, checkpoint_root = make_fake_project(
                root,
                track_grid=True,
                condition_mode=CONDITION_MODE_APPEARANCE_TRACK,
            )

            run_dir = train(config_path, run_id="patch_condition", max_steps=1, limit_samples=2)
            output_dir = infer(
                config_path,
                checkpoint_path=checkpoint_root / "patch_condition" / "best.pt",
                run_id="patch_condition",
                limit_samples=2,
                overwrite=True,
            )
            frame_scores = read_jsonl(output_dir / "frame_scores.jsonl")
            metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            checkpoint = torch.load(
                run_dir / "best.pt",
                map_location="cpu",
                weights_only=False,
            )

        self.assertTrue(frame_scores)
        self.assertEqual(metadata["track_condition"]["condition_mode"], "appearance_track")
        self.assertTrue(metadata["track_condition"]["patch_condition"])
        self.assertIn("patch_condition_adapter.proj.weight", checkpoint["model_state_dict"])
        self.assertNotIn("track_grid_gate", checkpoint["model_state_dict"])
        self.assertNotIn("high_adapter.proj.weight", checkpoint["model_state_dict"])

    def test_train_and_infer_smoke_with_track_grid_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, checkpoint_root = make_fake_project(root, track_grid=True)

            run_dir = train(config_path, run_id="track", max_steps=1, limit_samples=2)
            output_dir = infer(
                config_path,
                checkpoint_path=checkpoint_root / "track" / "best.pt",
                run_id="track",
                limit_samples=2,
                overwrite=True,
            )
            frame_scores = read_jsonl(output_dir / "frame_scores.jsonl")
            metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            checkpoint = torch.load(
                run_dir / "best.pt",
                map_location="cpu",
                weights_only=False,
            )

        self.assertTrue(frame_scores)
        self.assertTrue(metadata["track_condition"]["enabled"])
        self.assertEqual(metadata["track_condition"]["context_token_shape"], [1, 12, 16])
        self.assertIn("track_grid_gate", checkpoint["model_state_dict"])
        self.assertIn("track_grid_adapter.value_proj.weight", checkpoint["model_state_dict"])

    def test_train_initializes_from_checkpoint_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, checkpoint_root = make_fake_project(root)

            base_dir = train(config_path, run_id="base", max_steps=1, limit_samples=2)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["training"]["learning_rate"] = 0.0
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            init_checkpoint = checkpoint_root / "base" / "best.pt"

            ft_dir = train(
                config_path,
                run_id="ft",
                init_checkpoint_path=init_checkpoint,
                max_steps=1,
                limit_samples=2,
            )

            base_state = torch.load(base_dir / "best.pt", map_location="cpu", weights_only=False)[
                "model_state_dict"
            ]
            ft_checkpoint = torch.load(ft_dir / "best.pt", map_location="cpu", weights_only=False)
            metadata = json.loads((ft_dir / "run_metadata.json").read_text(encoding="utf-8"))

        for key, value in base_state.items():
            self.assertTrue(torch.equal(value, ft_checkpoint["model_state_dict"][key]))
        self.assertEqual(metadata["init_checkpoint_path"], str(init_checkpoint))

    def test_track_grid_train_initializes_from_baseline_checkpoint_strict_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, checkpoint_root = make_fake_project(root)

            base_dir = train(config_path, run_id="base", max_steps=1, limit_samples=2)
            track_config_path, _ = make_fake_project(root, track_grid=True)
            config = yaml.safe_load(track_config_path.read_text(encoding="utf-8"))
            config["training"]["learning_rate"] = 0.0
            track_config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            init_checkpoint = checkpoint_root / "base" / "best.pt"

            ft_dir = train(
                track_config_path,
                run_id="track_ft",
                init_checkpoint_path=init_checkpoint,
                max_steps=1,
                limit_samples=2,
            )

            base_state = torch.load(base_dir / "best.pt", map_location="cpu", weights_only=False)[
                "model_state_dict"
            ]
            ft_checkpoint = torch.load(ft_dir / "best.pt", map_location="cpu", weights_only=False)
            ft_state = ft_checkpoint["model_state_dict"]
            metadata = json.loads((ft_dir / "run_metadata.json").read_text(encoding="utf-8"))

        for key, value in base_state.items():
            self.assertTrue(torch.equal(value, ft_state[key]))
        self.assertTrue(metadata["track_condition"]["enabled"])
        self.assertEqual(metadata["init_checkpoint_path"], str(init_checkpoint))
        self.assertIn("track_grid_gate", ft_state)
        self.assertIn("track_grid_adapter.value_proj.weight", ft_state)


def make_fake_project(
    root: Path,
    *,
    track_grid: bool = False,
    condition_mode: str = "baseline",
) -> tuple[Path, Path]:
    features = root / "features"
    checkpoints = root / "checkpoints"
    predictions = root / "predictions"
    train_records = make_fake_feature_records(
        features,
        "train",
        count=3,
        labels=[0, 0, 0],
        track_grid=track_grid,
    )
    test_records = make_fake_feature_records(
        features,
        "test",
        count=2,
        labels=[0, 1],
        track_grid=track_grid,
    )
    write_jsonl(features / "train_feature_index.jsonl", train_records)
    write_jsonl(features / "test_feature_index.jsonl", test_records)

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
            "condition": {
                "mode": condition_mode,
                "use_track_grid": track_grid,
                "track_grid_gate": 0.1,
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
        "loss": {
            "alpha": 0.2,
            "topk_fraction": 0.1,
        },
        "optimization": {"matmul_precision": "high"},
        "training": {
            "feature_index": str(features / "train_feature_index.jsonl"),
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
            "feature_index": str(features / "test_feature_index.jsonl"),
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
            "variants": [
                "global",
                "patch",
                "low_weighted",
                "global_patch",
                "motion_patch",
                "motion_global_patch",
            ],
            "motion_topk_fraction": 0.25,
            "sweep": {"betas": [0.5], "topk_fractions": [0.2]},
            "score_normalization": {
                "enabled": True,
                "primary": "video_centered",
                "rolling_window": 128,
                "rolling_min_history": 16,
                "rolling_windows": [64],
            },
            "normalized_z_mse": {"eps": 1.0e-6},
            "decoded_mse": {"enabled": False, "alpha": 0.1},
        },
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, checkpoints


def run_patch_condition_forward(mode: str):
    shape = ZDiTShape(
        high_frames=3,
        high_tokens=5,
        high_dim=6,
        future_frames=2,
        z_channels=2,
        z_height=4,
        z_width=4,
        low_frames=2,
        low_channels=3,
        low_height=16,
        low_width=16,
        track_frames=3,
        track_channels=len(TRACK_CHANNELS),
        track_grid_h=2,
        track_grid_w=2,
    )
    model = ConditionedZDiT(
        shape=shape,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        high_max_tokens=8,
        high_use_pos_embedding=True,
        high_token_reduction="uniform",
        high_foreground_ratio=0.5,
        z_patch_size=2,
        z_use_temporal_pos=True,
        z_use_spatial_pos=True,
        low_patch_size=8,
        low_use_temporal_pos=True,
        low_use_spatial_pos=True,
        condition_mode=mode,
    )
    high = torch.randn(2, 3, 5, 6)
    low = torch.randn(2, 2, 3, 16, 16)
    track_grid = torch.randn(2, 3, len(TRACK_CHANNELS), 2, 2)
    z = torch.randn(2, 2, 2, 4, 4)
    time_values = torch.rand(2)

    prediction = model(z, time_values, high, low, track_grid)
    return prediction, model, z


def make_fake_feature_records(
    features: Path,
    split: str,
    *,
    count: int,
    labels: list[int],
    track_grid: bool = False,
) -> list[dict[str, object]]:
    records = []
    for index in range(count):
        sample_id = f"{split}_{index:03d}"
        high_path = features / "high" / split / f"{sample_id}.pt"
        z_path = features / "z" / split / f"{sample_id}.pt"
        low_path = features / "low" / split / f"{sample_id}.pt"
        track_path = features / "track" / split / f"{sample_id}.pt"
        high_path.parent.mkdir(parents=True, exist_ok=True)
        z_path.parent.mkdir(parents=True, exist_ok=True)
        low_path.parent.mkdir(parents=True, exist_ok=True)
        track_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sample_id": sample_id,
                "high": torch.randn(3, 5, 6),
            },
            high_path,
        )
        torch.save(
            {
                "sample_id": sample_id,
                "z": torch.randn(2, 2, 4, 4),
            },
            z_path,
        )
        torch.save(
            {
                "sample_id": sample_id,
                "low": torch.randn(2, 3, 16, 16),
            },
            low_path,
        )
        if track_grid:
            torch.save(
                {
                    "sample_id": sample_id,
                    "video_id": "video",
                    "context_grid": make_fake_track_grid(3, offset=index),
                    "future_grid": make_fake_track_grid(2, offset=index + 10),
                    "metadata": {
                        "grid_size": [2, 2],
                        "channels": list(TRACK_CHANNELS),
                        "trajectory_mode": "causal",
                    },
                },
                track_path,
            )
        records.append(
            {
                "sample_id": sample_id,
                "video_id": "video",
                "split": split,
                "scene_id": "scene",
                "future_frames": [index + 1, index + 2],
                "future_frame_labels": [0, int(labels[index])],
                "future_label": int(labels[index]),
                "high_feature_path": str(high_path),
                "z_path": str(z_path),
                "low_feature_path": str(low_path),
            }
        )
        if track_grid:
            records[-1]["track_feature_path"] = str(track_path)
    return records


def make_fake_track_grid(frames: int, *, offset: int) -> torch.Tensor:
    grid = torch.zeros(frames, len(TRACK_CHANNELS), 2, 2)
    for frame_idx in range(frames):
        row = (frame_idx + offset) % 2
        col = (frame_idx + offset // 2) % 2
        grid[frame_idx, 0, row, col] = 1.0
        grid[frame_idx, 1, row, col] = 0.1
        grid[frame_idx, 3, row, col] = 0.1
        grid[frame_idx, 4, row, col] = 1.0
    return grid


if __name__ == "__main__":
    unittest.main()
