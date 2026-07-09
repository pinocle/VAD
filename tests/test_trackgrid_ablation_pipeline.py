import tempfile
import unittest
from pathlib import Path

import torch
from src.models.z_dit import ConditionedZDiT, ZDiTShape
from src.pipelines.object_track_pipeline import TRACK_CHANNELS
from src.pipelines.trackgrid_ablation_pipeline import (
    AblationSpec,
    collect_summary_row,
    prepare_ablation_config,
    write_summary,
)
from src.pipelines.z_dit_pipeline import (
    SCORE_VARIANT_MEAN_TOPK_PLUS_TRACK_REGION,
    SCORE_VARIANT_TRACK_WEIGHTED,
    apply_track_grid_ablation,
    apply_track_grid_channel_mask,
    apply_track_grid_gate_override,
    load_pipeline_config,
)


class TrackGridAblationPipelineTests(unittest.TestCase):
    def test_zero_track_produces_zero_grid_with_same_shape(self):
        grid = torch.randn(2, 3, len(TRACK_CHANNELS), 4, 5)

        ablated = apply_track_grid_ablation(grid, "zero_track", seed=7)

        self.assertEqual(tuple(ablated.shape), tuple(grid.shape))
        self.assertEqual(float(ablated.abs().sum()), 0.0)

    def test_shuffled_track_is_deterministic_for_fixed_seed(self):
        grid = torch.arange(4 * 2 * len(TRACK_CHANNELS) * 2 * 2, dtype=torch.float32).reshape(
            4,
            2,
            len(TRACK_CHANNELS),
            2,
            2,
        )

        first = apply_track_grid_ablation(grid, "shuffled_track", seed=13)
        second = apply_track_grid_ablation(grid, "shuffled_track", seed=13)

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, grid))
        self.assertEqual(float(first.sum()), float(grid.sum()))

    def test_channel_mask_keeps_only_selected_channels(self):
        grid = torch.ones(1, 2, len(TRACK_CHANNELS), 2, 2)

        masked = apply_track_grid_channel_mask(grid, "objectness_speed")

        kept = {TRACK_CHANNELS.index("objectness"), TRACK_CHANNELS.index("speed")}
        for channel_index in range(len(TRACK_CHANNELS)):
            channel_sum = float(masked[:, :, channel_index].sum())
            if channel_index in kept:
                self.assertGreater(channel_sum, 0.0)
            else:
                self.assertEqual(channel_sum, 0.0)

    def test_gate_override_changes_track_grid_gate_value(self):
        model = make_track_grid_model()

        gate_value = apply_track_grid_gate_override(model, 0.25)

        self.assertAlmostEqual(gate_value, 0.25)
        self.assertAlmostEqual(float(model.track_grid_gate.detach()), 0.25)

    def test_existing_inference_config_defaults_to_no_ablation_behavior(self):
        config = load_pipeline_config(Path("config/local.yaml"))

        self.assertFalse(config.model.condition.use_track_grid)
        self.assertEqual(config.model.condition.track_grid.ablation_mode, "real")
        self.assertEqual(config.model.condition.track_grid.channel_mask, "all_channels")
        self.assertIsNone(config.model.condition.track_grid.gate_override)

    def test_prepare_ablation_config_sets_track_weighted_primary_variant(self):
        raw = {
            "model": {"condition": {"use_track_grid": True}},
            "inference": {"feature_index": "old.jsonl"},
            "scoring": {
                "variant": SCORE_VARIANT_MEAN_TOPK_PLUS_TRACK_REGION,
                "variants": [SCORE_VARIANT_MEAN_TOPK_PLUS_TRACK_REGION],
            },
        }

        config = prepare_ablation_config(
            raw,
            spec=AblationSpec(
                ablation_mode="zero",
                channel_mask="speed_only",
                gate_override=0.0,
            ),
            feature_index=Path("features_tracks.jsonl"),
            shuffle_seed=3,
        )

        self.assertEqual(config["scoring"]["variant"], SCORE_VARIANT_TRACK_WEIGHTED)
        self.assertIn(SCORE_VARIANT_TRACK_WEIGHTED, config["scoring"]["variants"])
        self.assertIn(SCORE_VARIANT_MEAN_TOPK_PLUS_TRACK_REGION, config["scoring"]["variants"])
        self.assertEqual(config["model"]["condition"]["track_grid"]["ablation_mode"], "zero")
        self.assertEqual(config["model"]["condition"]["track_grid"]["channel_mask"], "speed_only")
        self.assertEqual(config["model"]["condition"]["track_grid"]["gate_override"], 0.0)
        self.assertEqual(config["inference"]["feature_index"], "features_tracks.jsonl")

    def test_summary_csv_and_json_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "predictions" / "run"
            output_dir.mkdir(parents=True)
            metrics = {
                "track_grid_ablation": {"gate_value": 0.1},
                "score_metrics": {
                    "global_patch_video_centered_score": {
                        "roc_auc": 0.7,
                        "average_precision": 0.4,
                    },
                    "track_weighted_video_centered_score": {
                        "roc_auc": 0.8,
                        "average_precision": 0.5,
                    },
                    "global_patch_rolling_centered_score": {
                        "roc_auc": 0.6,
                        "average_precision": 0.3,
                    },
                    "track_weighted_rolling_centered_score": {
                        "roc_auc": 0.65,
                        "average_precision": 0.35,
                    },
                    "score": {"roc_auc": 0.8, "average_precision": 0.5},
                },
            }
            (output_dir / "metrics.json").write_text(
                __import__("json").dumps(metrics),
                encoding="utf-8",
            )

            row = collect_summary_row(
                output_dir,
                run_id="run",
                checkpoint_path=Path("checkpoints/run/best.pt"),
                spec=AblationSpec("real", "all_channels", None),
            )
            write_summary(root / "summary", [row])

            self.assertTrue((root / "summary" / "summary.csv").is_file())
            self.assertTrue((root / "summary" / "summary.json").is_file())
            self.assertEqual(row["track_weighted_video_centered_auc"], 0.8)


def make_track_grid_model() -> ConditionedZDiT:
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
    return ConditionedZDiT(
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


if __name__ == "__main__":
    unittest.main()
