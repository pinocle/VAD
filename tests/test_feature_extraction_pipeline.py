import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from PIL import Image
from src.pipelines.feature_extraction_pipeline import (
    autocast_is_enabled,
    build_feature_index_records,
    compute_low_tensor,
    high_cache_path,
    high_signature,
    load_config,
    load_low_frame_tensor,
    load_vae_frame_tensor,
    low_cache_path,
    low_signature,
    model_load_kwargs,
    pending_high_records,
    select_hidden_tokens,
    select_high_layer,
    select_vae_z,
    torch_dtype,
    z_cache_path,
    z_signature,
)


class FeatureExtractionPipelineTests(unittest.TestCase):
    def test_load_config_resolves_common_and_local_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset": {
                            "processed_root": str(root / "processed"),
                        },
                        "features": {
                            "root": str(root / "features"),
                            "batch_size": 16,
                            "num_workers": 4,
                            "dtype": "float16",
                            "high": {
                                "type": "image",
                                "model_id": "facebook/dinov2-base",
                                "image_size": 224,
                                "processor_do_resize": False,
                                "processor_do_center_crop": False,
                                "output_layer": -2,
                                "token_mode": "patch",
                                "batch_size": 32,
                                "freeze": True,
                                "cache": True,
                            },
                            "z": {
                                "type": "vae",
                                "model_id": "madebyollin/sdxl-vae-fp16-fix",
                                "image_size": 256,
                                "z_mode": "mode",
                                "freeze": True,
                                "cache": True,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.high.batch_size, 32)
        self.assertFalse(config.high.processor_do_resize)
        self.assertFalse(config.high.processor_do_center_crop)
        self.assertEqual(config.z.batch_size, 16)
        self.assertEqual(config.low.batch_size, 16)
        self.assertEqual(config.high.num_workers, 4)
        self.assertEqual(config.z.dtype, "float16")
        self.assertEqual(config.low.mode, "signed")
        self.assertEqual(config.low.method, "farneback")

    def test_load_config_rejects_cache_when_encoder_is_not_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config = {
                "dataset": {"processed_root": str(root / "processed")},
                "features": {
                    "high": {
                        "model_id": "facebook/dinov2-base",
                        "freeze": False,
                        "cache": True,
                    },
                    "z": {"model_id": "madebyollin/sdxl-vae-fp16-fix"},
                },
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "high.cache"):
                load_config(config_path)

            config["features"]["high"]["freeze"] = True
            config["features"]["z"]["freeze"] = False
            config["features"]["z"]["cache"] = True
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "z.cache"):
                load_config(config_path)

    def test_load_config_rejects_invalid_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config = {
                "dataset": {"processed_root": str(root / "processed")},
                "features": {
                    "high": {
                        "model_id": "facebook/dinov2-base",
                        "token_mode": "tokens",
                    },
                    "z": {"model_id": "madebyollin/sdxl-vae-fp16-fix"},
                },
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "token_mode"):
                load_config(config_path)

            config["features"]["high"]["token_mode"] = "patch"
            config["features"]["z"]["z_mode"] = "posterior"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "z_mode"):
                load_config(config_path)

            config["features"]["z"]["z_mode"] = "mode"
            config["features"]["low"] = {"mode": "magnitude"}
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "low.mode"):
                load_config(config_path)

            config["features"]["low"] = {"type": "optical_flow", "mode": "signed"}
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "low.mode"):
                load_config(config_path)

    def test_torch_dtype_mapping(self):
        self.assertEqual(torch_dtype("float16"), torch.float16)
        self.assertEqual(torch_dtype("bfloat16"), torch.bfloat16)
        self.assertEqual(torch_dtype("float32"), torch.float32)
        with self.assertRaises(ValueError):
            torch_dtype("float64")

    def test_model_dtype_optimization_is_cuda_only(self):
        self.assertEqual(model_load_kwargs("float16", torch.device("cpu")), {})
        self.assertEqual(
            model_load_kwargs("float16", torch.device("cuda")),
            {"torch_dtype": torch.float16},
        )
        self.assertTrue(autocast_is_enabled(torch.float16, torch.device("cuda")))
        self.assertTrue(autocast_is_enabled(torch.bfloat16, torch.device("cuda")))
        self.assertFalse(autocast_is_enabled(torch.float32, torch.device("cuda")))
        self.assertFalse(autocast_is_enabled(torch.float16, torch.device("cpu")))

    def test_select_hidden_tokens_modes(self):
        hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)

        self.assertTrue(torch.equal(select_hidden_tokens(hidden, "cls"), hidden[:, :1]))
        self.assertTrue(torch.equal(select_hidden_tokens(hidden, "patch"), hidden[:, 1:]))
        self.assertTrue(torch.equal(select_hidden_tokens(hidden, "all"), hidden))

        with self.assertRaises(ValueError):
            select_hidden_tokens(hidden[:, :1], "patch")

    def test_select_high_layer_uses_hidden_states(self):
        hidden_states = (
            torch.full((1, 2, 3), 1.0),
            torch.full((1, 2, 3), 2.0),
            torch.full((1, 2, 3), 3.0),
        )
        outputs = SimpleNamespace(hidden_states=hidden_states)

        self.assertTrue(torch.equal(select_high_layer(outputs, -2), hidden_states[1]))

        with self.assertRaises(ValueError):
            select_high_layer(SimpleNamespace(), -1)
        with self.assertRaises(ValueError):
            select_high_layer(outputs, 5)

    def test_select_vae_z_modes(self):
        distribution = SimpleNamespace(
            mean=torch.ones(1, 2),
            mode=lambda: torch.full((1, 2), 2.0),
            sample=lambda: torch.full((1, 2), 3.0),
        )

        self.assertTrue(torch.equal(select_vae_z(distribution, "mean"), distribution.mean))
        self.assertTrue(torch.equal(select_vae_z(distribution, "mode"), torch.full((1, 2), 2.0)))
        self.assertTrue(torch.equal(select_vae_z(distribution, "sample"), torch.full((1, 2), 3.0)))

    def test_load_vae_frame_tensor_resizes_and_normalizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame_path = Path(tmp) / "frame.png"
            Image.new("RGB", (2, 2), color=(255, 0, 127)).save(frame_path)

            tensor = load_vae_frame_tensor(str(frame_path), image_size=4)

        self.assertEqual(tuple(tensor.shape), (3, 4, 4))
        self.assertLessEqual(float(tensor.max()), 1.0)
        self.assertGreaterEqual(float(tensor.min()), -1.0)
        self.assertAlmostEqual(float(tensor[0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(tensor[1, 0, 0]), -1.0)

    def test_compute_low_tensor_from_past_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "frame_001.png"
            second = Path(tmp) / "frame_002.png"
            third = Path(tmp) / "frame_003.png"
            Image.new("RGB", (2, 2), color=(255, 0, 0)).save(first)
            Image.new("RGB", (2, 2), color=(0, 0, 0)).save(second)
            Image.new("RGB", (2, 2), color=(0, 255, 0)).save(third)
            frames = torch.stack(
                [
                    load_low_frame_tensor(str(first), image_size=4),
                    load_low_frame_tensor(str(second), image_size=4),
                    load_low_frame_tensor(str(third), image_size=4),
                ]
            )

            signed = compute_low_tensor(frames, "signed")
            magnitude = compute_low_tensor(frames, "abs")

        self.assertEqual(tuple(signed.shape), (2, 3, 4, 4))
        self.assertGreater(float(signed[0, 0].mean()), 0.9)
        self.assertLess(float(signed[1, 1].mean()), -0.9)
        self.assertGreaterEqual(float(magnitude.min()), 0.0)

    def test_compute_optical_flow_low_tensor(self):
        frames = torch.zeros(2, 3, 32, 32)
        frames[0, :, 8:20, 8:20] = 1.0
        frames[1, :, 8:20, 12:24] = 1.0

        flow = compute_low_tensor(
            frames,
            "uv_mag",
            low_type="optical_flow",
            method="farneback",
        )

        self.assertEqual(tuple(flow.shape), (1, 3, 32, 32))
        self.assertGreaterEqual(float(flow[:, 2].min()), 0.0)
        self.assertGreater(float(flow[:, 2].max()), 0.0)

    def test_signatures_paths_and_index_records_are_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset": {"processed_root": str(root / "processed")},
                        "features": {
                            "root": str(root / "features"),
                            "high": {"model_id": "facebook/dinov2-base"},
                            "z": {"model_id": "madebyollin/sdxl-vae-fp16-fix"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(config_path)
            samples = [
                {
                    "sample_id": "sample_001",
                    "video_id": "video",
                    "split": "train",
                    "future_label": 0,
                }
            ]

            index_records = build_feature_index_records(
                config,
                "train",
                samples,
                limit_samples=None,
            )

        self.assertIn("facebook_dinov2-base", high_signature(config.high))
        self.assertIn("madebyollin_sdxl-vae-fp16-fix", z_signature(config.z))
        self.assertIn("frame_diff", low_signature(config.low))
        self.assertEqual(
            index_records[0]["high_feature_path"],
            str(high_cache_path(config, "train", "sample_001")),
        )
        self.assertEqual(
            index_records[0]["z_path"],
            str(z_cache_path(config, "train", "sample_001")),
        )
        self.assertEqual(
            index_records[0]["low_feature_path"],
            str(low_cache_path(config, "train", "sample_001")),
        )

    def test_pending_high_records_skips_existing_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset": {"processed_root": str(root / "processed")},
                        "features": {
                            "root": str(root / "features"),
                            "high": {"model_id": "facebook/dinov2-base"},
                            "z": {"model_id": "madebyollin/sdxl-vae-fp16-fix"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(config_path)
            samples = [{"sample_id": "cached"}, {"sample_id": "missing"}]
            cached_path = high_cache_path(config, "train", "cached")
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            cached_path.touch()

            pending = pending_high_records(config, "train", samples, overwrite=False)
            overwritten = pending_high_records(config, "train", samples, overwrite=True)

        self.assertEqual([sample["sample_id"] for sample in pending], ["missing"])
        self.assertEqual([sample["sample_id"] for sample in overwritten], ["cached", "missing"])


if __name__ == "__main__":
    unittest.main()
