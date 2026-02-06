from __future__ import annotations

import os
from pathlib import Path

from ragprep.config import Settings


def _set_env_if_missing(name: str, value: str) -> None:
    existing = os.getenv(name)
    if existing is not None and existing.strip():
        return
    os.environ[name] = value


def configure_model_cache(settings: Settings) -> Path:
    """
    Configure and materialize model cache directories shared by prewarm/runtime paths.

    This reduces repeated model downloads and keeps cache locations stable across runs.
    """

    root = Path(settings.model_cache_dir).expanduser()
    hf_root = root / "huggingface"
    hf_hub = hf_root / "hub"
    hf_transformers = hf_root / "transformers"
    torch_root = root / "torch"
    paddle_root = root / "paddle"
    paddlex_root = root / "paddlex"

    for directory in (
        root,
        hf_root,
        hf_hub,
        hf_transformers,
        torch_root,
        paddle_root,
        paddlex_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    _set_env_if_missing("HF_HOME", str(hf_root))
    _set_env_if_missing("HUGGINGFACE_HUB_CACHE", str(hf_hub))
    _set_env_if_missing("TRANSFORMERS_CACHE", str(hf_transformers))
    _set_env_if_missing("TORCH_HOME", str(torch_root))
    _set_env_if_missing("PADDLE_HOME", str(paddle_root))
    _set_env_if_missing("PADDLEX_HOME", str(paddlex_root))

    return root
