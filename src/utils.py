"""Shared project utilities."""

from __future__ import annotations

import gc
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
T = TypeVar("T")


@dataclass(frozen=True)
class ProgressConfig:
    """Shared tqdm display settings."""

    enabled: bool = True
    mininterval: float = 5.0
    dynamic_ncols: bool = True
    smoothing: float = 0.05


_PROGRESS_CONFIG = ProgressConfig()


def resolve_project_path(path: Path | str) -> Path:
    """Resolve a path relative to the project root when needed."""

    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def configure_progress_from_config(config: Mapping[str, Any]) -> ProgressConfig:
    """Configure shared progress bars from a top-level YAML config."""

    logging_config = config.get("logging", {})
    if logging_config is None:
        logging_config = {}
    if not isinstance(logging_config, Mapping):
        raise ValueError("logging must be a mapping")

    progress_config = logging_config.get("progress", {})
    if progress_config is None:
        progress_config = {}
    if not isinstance(progress_config, Mapping):
        raise ValueError("logging.progress must be a mapping")
    return configure_progress(progress_config)


def configure_progress(config: Mapping[str, Any] | None = None) -> ProgressConfig:
    """Set shared progress bar defaults."""

    global _PROGRESS_CONFIG

    config = config or {}
    enabled = bool(config.get("enabled", True))
    mininterval = float(config.get("mininterval", 5.0))
    dynamic_ncols = bool(config.get("dynamic_ncols", True))
    smoothing = float(config.get("smoothing", 0.05))

    env_disable = parse_optional_env_bool("TQDM_DISABLE")
    if env_disable is not None:
        enabled = not env_disable
    env_mininterval = os.getenv("TQDM_MININTERVAL")
    if env_mininterval:
        mininterval = float(env_mininterval)

    if mininterval < 0:
        raise ValueError("logging.progress.mininterval must be non-negative")
    if not 0 <= smoothing <= 1:
        raise ValueError("logging.progress.smoothing must be in [0, 1]")

    _PROGRESS_CONFIG = ProgressConfig(
        enabled=enabled,
        mininterval=mininterval,
        dynamic_ncols=dynamic_ncols,
        smoothing=smoothing,
    )
    return _PROGRESS_CONFIG


def progress_bar(iterable: Iterable[T] | None = None, **kwargs: Any) -> Any:
    """Create a tqdm progress bar with project defaults."""

    kwargs.setdefault("disable", not _PROGRESS_CONFIG.enabled)
    kwargs.setdefault("mininterval", _PROGRESS_CONFIG.mininterval)
    kwargs.setdefault("dynamic_ncols", _PROGRESS_CONFIG.dynamic_ncols)
    kwargs.setdefault("smoothing", _PROGRESS_CONFIG.smoothing)
    return tqdm(iterable, **kwargs)


def progress_write(message: str) -> None:
    """Write a message without corrupting active progress bars."""

    tqdm.write(message)


def cleanup_memory(*, cuda: bool = True) -> None:
    """Release Python objects and unused CUDA cache at coarse pipeline boundaries."""

    gc.collect()
    if not cuda:
        return
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()


def parse_optional_env_bool(name: str) -> bool | None:
    """Parse an optional boolean environment variable."""

    value = os.getenv(name)
    if value is None or value == "":
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")
