from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from ragprep.ocr import lightonocr


def test_dry_run_skips_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(lightonocr.ENV_DRY_RUN, "1")

    def should_not_load(*_args: object, **_kwargs: object) -> lightonocr._Runtime:
        raise AssertionError("Model load should be skipped in dry-run mode.")

    monkeypatch.setattr(lightonocr, "_get_runtime_cached", should_not_load)
    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    assert lightonocr.ocr_image(image) == lightonocr.DRY_RUN_OUTPUT


def test_runtime_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    lightonocr._get_runtime_cached.cache_clear()

    calls: dict[str, int] = {"n": 0}

    def fake_load_runtime(*, model_id: str, device: str, dtype: str | None) -> lightonocr._Runtime:
        calls["n"] += 1
        return lightonocr._Runtime(
            torch=SimpleNamespace(float32=object(), bfloat16=object(), float16=object()),
            model=object(),
            processor=object(),
            device=device,
            dtype=object(),
        )

    monkeypatch.setattr(lightonocr, "_load_runtime", fake_load_runtime)

    runtime_1 = lightonocr._get_runtime_cached("m", "cpu", None)
    runtime_2 = lightonocr._get_runtime_cached("m", "cpu", None)
    assert runtime_1 is runtime_2
    assert calls["n"] == 1


def test_device_default_is_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(lightonocr.ENV_DEVICE, raising=False)
    settings = lightonocr.get_settings()
    assert settings.backend == "transformers"
    assert settings.device == "cpu"


def test_backend_invalid_value_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(lightonocr.ENV_BACKEND, "nope")
    with pytest.raises(ValueError, match=lightonocr.ENV_BACKEND):
        lightonocr.get_settings()


def test_llama_cpp_requires_gguf_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(lightonocr.ENV_BACKEND, "llama_cpp")
    monkeypatch.delenv(lightonocr.ENV_GGUF_MODEL_PATH, raising=False)
    monkeypatch.delenv(lightonocr.ENV_GGUF_MMPROJ_PATH, raising=False)

    with pytest.raises(ValueError) as excinfo:
        lightonocr.get_settings()

    message = str(excinfo.value)
    assert lightonocr.ENV_GGUF_MODEL_PATH in message
    assert lightonocr.ENV_GGUF_MMPROJ_PATH in message


def test_dry_run_skips_settings_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(lightonocr.ENV_DRY_RUN, "1")
    monkeypatch.setenv(lightonocr.ENV_BACKEND, "llama_cpp")
    monkeypatch.delenv(lightonocr.ENV_GGUF_MODEL_PATH, raising=False)
    monkeypatch.delenv(lightonocr.ENV_GGUF_MMPROJ_PATH, raising=False)

    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    assert lightonocr.ocr_image(image) == lightonocr.DRY_RUN_OUTPUT


def test_load_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    lightonocr._get_runtime_cached.cache_clear()

    def fake_load_runtime(*, model_id: str, device: str, dtype: str | None) -> lightonocr._Runtime:
        raise ImportError("boom")

    monkeypatch.setattr(lightonocr, "_load_runtime", fake_load_runtime)

    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    with pytest.raises(RuntimeError, match="Failed to load LightOnOCR"):
        lightonocr.ocr_image(image)
