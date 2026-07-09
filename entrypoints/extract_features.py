"""Run the configured C_high/C_low/Z feature extraction pipeline."""

# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.feature_extraction_pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()
