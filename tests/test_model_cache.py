from __future__ import annotations

import os
from pathlib import Path

import pytest

from ragprep import config
from ragprep.model_cache import configure_model_cache


def test_configure_model_cache_creates_expected_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RAGPREP_MODEL_CACHE_DIR", str(tmp_path / "model-cache"))
    settings = config.get_settings()

    root = configure_model_cache(settings)
    assert root.exists()
    assert (root / "huggingface").is_dir()
    assert (root / "huggingface" / "hub").is_dir()
    assert (root / "huggingface" / "transformers").is_dir()
    assert (root / "torch").is_dir()
    assert (root / "paddle").is_dir()
    assert (root / "paddlex").is_dir()


def test_configure_model_cache_sets_env_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RAGPREP_MODEL_CACHE_DIR", str(tmp_path / "model-cache"))
    for key in (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "PADDLE_HOME",
        "PADDLEX_HOME",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = config.get_settings()
    root = configure_model_cache(settings)

    assert os.environ["HF_HOME"] == str(root / "huggingface")
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(root / "huggingface" / "hub")
    assert os.environ["TRANSFORMERS_CACHE"] == str(root / "huggingface" / "transformers")
    assert os.environ["TORCH_HOME"] == str(root / "torch")
    assert os.environ["PADDLE_HOME"] == str(root / "paddle")
    assert os.environ["PADDLEX_HOME"] == str(root / "paddlex")


def test_configure_model_cache_preserves_existing_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RAGPREP_MODEL_CACHE_DIR", str(tmp_path / "model-cache"))
    monkeypatch.setenv("HF_HOME", "C:/existing-hf-home")

    settings = config.get_settings()
    _ = configure_model_cache(settings)

    assert os.environ["HF_HOME"] == "C:/existing-hf-home"

