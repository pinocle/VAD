import unittest
from pathlib import Path

from src.pipelines.feature_eng_pipeline import load_config as load_preprocess_config
from src.pipelines.feature_extraction_pipeline import load_config as load_feature_config
from src.pipelines.z_dit_pipeline import load_pipeline_config


class PipelineConnectivityTests(unittest.TestCase):
    def test_local_config_connects_all_pipeline_stages(self):
        config_path = Path("config/local.yaml")

        preprocess_config = load_preprocess_config(config_path)
        feature_config = load_feature_config(config_path)
        z_config = load_pipeline_config(config_path)

        self.assertEqual(feature_config.processed_root, preprocess_config.processed_root)
        self.assertEqual(
            z_config.training.feature_index,
            feature_config.feature_index_path("train"),
        )
        self.assertEqual(
            z_config.inference.feature_index,
            feature_config.feature_index_path("test"),
        )
        self.assertEqual(
            z_config.inference.output_root,
            Path("data/04_predictions") / z_config.dataset_name,
        )
        self.assertEqual(feature_config.high.image_size // 14, 16)
        self.assertEqual(
            feature_config.z.image_size // 8 // z_config.model.z_adapter.patch_size, 16
        )
        self.assertEqual(
            feature_config.low.image_size // z_config.model.low_adapter.patch_size,
            16,
        )


if __name__ == "__main__":
    unittest.main()
