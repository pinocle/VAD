## 프로젝트 구조
파일명은 그대로 따를 것. 답변은 항상 한국어로 작성할 것.

/workspace/VAD
├── checkpoints/
├── config/
|   ├── local.yaml
|   └── prod.yaml
├── data/
|   ├── 01_raw/
|   ├── 02_processed/
|   ├── 03_features/
|   └── 04_predictions/
├── entrypoints/
|   ├── inference.py
|   └── train.py
├── notebooks/
|   ├── EDA.ipynb
|   └── Baseline.ipynb
├── src/
|   ├── models/
|   ├── pipelines/
|   |   ├── __init__.py
|   |   ├── feature_eng_pipeline.py
|   |   ├── inference_pipeline.py
|   |   └── training_pipeline.py
|   └── utils.py
├── tests/
|   ├── __init__.py
|   └── test_training.py
├── docker-compose.yaml(compose.yaml)
├── Dockerfile
├── env.yaml
├── env-dev.yaml
├── README.md
├── requirements.txt
└── requirements-dev.txt

- checkpoints/ -> model checkpoints
    Save best model weights for inference
- config/ -> config files
    Separate params from code. (local.yaml, prod.yaml)
- data/ -> full data lifecycle
    raw->preprocessed->features->predictions
- entrypoints/ -> main scripts
    train.py (pipeline)
    inference.py (batch/real-time)
- notebooks/ -> exploration only
    EDA, analysis - never production logic
- models/ -> model definitions
    model architecture
- src/ -> core ML code
    feature engineering, training, inference (modular + testable)
- tests/ -> automated checks
    prevent silent failures
- docker + env files -> reproducibility
    same setup on any machine/CI
- pinned dependencies -> stability
    exact versions -> consistent results

## Code style
Formatter: ruff format (double quotes, spaces, line-length 100)
Linter: ruff check with rules E, F, I (ignore E501)
Config lives in pyproject.toml under [tool.ruff]