import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from src.pipelines.rgb_patch_pipeline import (
    RGBPatchDataset,
    RGBPatchModelConfig,
    ScoringConfig,
    patchify_rgb,
    score_patch_prediction,
    unpatchify_rgb,
)


class FixedGridRGBPatchTests(unittest.TestCase):
    def test_patchify_unpatchify_is_exact_for_supported_sizes(self) -> None:
        for image_size, patch_size in ((32, 16), (64, 32)):
            images = torch.arange(
                2 * 3 * 3 * image_size * image_size,
                dtype=torch.float32,
            ).reshape(2, 3, 3, image_size, image_size)
            patches = patchify_rgb(images, patch_size)
            grid_size = image_size // patch_size
            self.assertEqual(
                tuple(patches.shape),
                (2, 3, grid_size * grid_size, 3 * patch_size * patch_size),
            )
            restored = unpatchify_rgb(
                patches,
                image_size=image_size,
                patch_size=patch_size,
            )
            self.assertTrue(torch.equal(restored, images))

    def test_patchify_uses_row_major_fixed_grid_order(self) -> None:
        image = torch.zeros(1, 3, 32, 32)
        for row in range(2):
            for column in range(2):
                image[:, :, row * 16 : (row + 1) * 16, column * 16 : (column + 1) * 16] = (
                    row * 2 + column
                )
        patches = patchify_rgb(image, 16)
        patch_values = patches[0].mean(dim=-1)
        self.assertTrue(torch.equal(patch_values, torch.tensor([0.0, 1.0, 2.0, 3.0])))

    def test_dataset_returns_context_and_future_fixed_grid_patches(self) -> None:
        model = RGBPatchModelConfig(32, 16, 32, 1, 4, 2.0, 0.0, 4, 0.25)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context_paths = []
            future_paths = []
            for index in range(5):
                path = root / f"{index}.png"
                Image.new("RGB", (20, 24), color=(index * 20, 100, 50)).save(path)
                (context_paths if index < 3 else future_paths).append(str(path))
            dataset = RGBPatchDataset(
                [
                    {
                        "sample_id": "sample",
                        "context_frame_paths": context_paths,
                        "future_frame_paths": future_paths,
                    }
                ],
                model,
            )
            sample = dataset[0]
        self.assertEqual(tuple(sample["context"].shape), (3, 4, 768))
        self.assertEqual(tuple(sample["target"].shape), (2, 4, 768))
        self.assertGreaterEqual(float(sample["context"].min()), -1.0)
        self.assertLessEqual(float(sample["target"].max()), 1.0)

    def test_score_combines_full_frame_topk_and_memory_components(self) -> None:
        target = torch.zeros(1, 1, 4, 3)
        prediction = torch.zeros_like(target)
        prediction[0, 0, 0] = 2.0
        prediction[0, 0, 1] = 1.0
        scores = score_patch_prediction(
            target,
            prediction,
            scoring=ScoringConfig(0.25, 0.5, 0.25),
            memory_distance=torch.tensor([0.5]),
        )
        self.assertAlmostEqual(float(scores["full_frame"][0, 0]), 1.25)
        self.assertAlmostEqual(float(scores["topk"][0, 0]), 4.0)
        self.assertAlmostEqual(float(scores["memory_distance"][0, 0]), 0.5)
        self.assertAlmostEqual(float(scores["score"][0, 0]), 3.375)


if __name__ == "__main__":
    unittest.main()
