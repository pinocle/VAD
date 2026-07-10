import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from PIL import Image
from src.models import FlowMatcher, RGBPatchDiT
from src.pipelines.inference_pipeline import infer
from src.pipelines.rgb_patch_pipeline import (
    RGBPatchModelConfig,
    RGBPatchShape,
    read_jsonl,
    write_jsonl,
)
from src.pipelines.training_pipeline import train


class RGBPatchDiTTests(unittest.TestCase):
    def test_cross_attention_memory_dit_and_flow_shapes(self) -> None:
        shape = RGBPatchShape(3, 2, 4, 768, 32, 16)
        model_config = RGBPatchModelConfig(32, 16, 32, 2, 4, 2.0, 0.0, 8, 0.25)
        model = RGBPatchDiT(shape=shape, config=model_config)
        flow = FlowMatcher(inference_steps=2)
        context = torch.randn(2, 3, 4, 768)
        target = torch.randn(2, 2, 4, 768)
        noisy_target, time_values, velocity_target = flow.prepare_training_pair(target)
        prediction, memory_distance = model(noisy_target, time_values, context)
        generated, sampled_memory = flow.sample(
            model,
            context=context,
            target_shape=tuple(target.shape),
        )
        self.assertEqual(tuple(prediction.shape), tuple(velocity_target.shape))
        self.assertEqual(tuple(generated.shape), tuple(target.shape))
        self.assertEqual(tuple(memory_distance.shape), (2,))
        self.assertEqual(tuple(sampled_memory.shape), (2,))

    def test_train_and_noise_inference_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, checkpoint_root = make_project(root)
            run_dir = train(config_path, run_id="smoke", max_steps=2, overwrite=True)
            output_dir = infer(
                config_path,
                checkpoint_path=checkpoint_root / "smoke" / "best.pt",
                run_id="smoke",
                overwrite=True,
            )
            frame_scores = read_jsonl(output_dir / "frame_scores.jsonl")
            predictions = read_jsonl(output_dir / "future_frame_predictions.jsonl")
            checkpoint_exists = (run_dir / "best.pt").is_file()
        self.assertTrue(checkpoint_exists)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(frame_scores), 4)
        self.assertIn("full_frame_score", frame_scores[0])
        self.assertIn("topk_score", frame_scores[0])
        self.assertIn("memory_distance", frame_scores[0])


def make_project(root: Path) -> tuple[Path, Path]:
    frames = root / "frames"
    frames.mkdir()
    train_records = make_records(frames, "train", 3, labels=[0, 0, 0])
    test_records = make_records(frames, "test", 2, labels=[0, 1])
    train_index = root / "samples_train.jsonl"
    test_index = root / "samples_test.jsonl"
    write_jsonl(train_index, train_records)
    write_jsonl(test_index, test_records)
    checkpoint_root = root / "checkpoints"
    config = {
        "dataset": {"name": "smoke"},
        "model": {
            "name": "rgb_patch_dit",
            "image_size": 32,
            "patch_size": 16,
            "hidden_size": 32,
            "num_layers": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "memory": {"size": 4, "temperature": 0.25},
        },
        "flow_matching": {"inference_steps": 2, "timestep_distribution": "uniform"},
        "training": {
            "sample_index": str(train_index),
            "output_root": str(checkpoint_root),
            "batch_size": 2,
            "num_workers": 0,
            "max_steps": 2,
            "val_fraction": 0.0,
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "dtype": "float32",
            "amp": False,
            "compile": False,
            "seed": 1,
            "log_every_steps": 1,
            "save_every_steps": 1,
        },
        "inference": {
            "sample_index": str(test_index),
            "output_root": str(root / "predictions"),
            "batch_size": 2,
            "num_workers": 0,
            "compile": False,
            "save_tensors": True,
            "overwrite": True,
        },
        "scoring": {
            "topk_fraction": 0.5,
            "topk_weight": 0.2,
            "memory_distance_weight": 0.1,
        },
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, checkpoint_root


def make_records(
    frames: Path,
    split: str,
    count: int,
    *,
    labels: list[int],
) -> list[dict[str, object]]:
    records = []
    for sample_index in range(count):
        context_paths = []
        future_paths = []
        for frame_index in range(4):
            path = frames / f"{split}_{sample_index}_{frame_index}.png"
            Image.new(
                "RGB",
                (24, 28),
                color=(sample_index * 30, frame_index * 40, 100),
            ).save(path)
            (context_paths if frame_index < 2 else future_paths).append(str(path))
        records.append(
            {
                "sample_id": f"{split}_{sample_index}",
                "video_id": "video",
                "future_label": labels[sample_index],
                "context_frame_paths": context_paths,
                "future_frame_paths": future_paths,
                "future_frames": [sample_index * 2 + 1, sample_index * 2 + 2],
                "future_frame_labels": [0, labels[sample_index]],
            }
        )
    return records


if __name__ == "__main__":
    unittest.main()
