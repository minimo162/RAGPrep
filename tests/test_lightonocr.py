from __future__ import annotations

import pytest
from PIL import Image

from ragprep.ocr import lightonocr


def test_dry_run_skips_settings_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(lightonocr.ENV_DRY_RUN, "1")
    monkeypatch.delenv(lightonocr.ENV_GGUF_MODEL_PATH, raising=False)
    monkeypatch.delenv(lightonocr.ENV_GGUF_MMPROJ_PATH, raising=False)

    def should_not_read_settings() -> lightonocr.LightOnOCRSettings:
        raise AssertionError("Settings should not be read in dry-run mode.")

    monkeypatch.setattr(lightonocr, "get_settings", should_not_read_settings)

    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    assert lightonocr.ocr_image(image) == lightonocr.DRY_RUN_OUTPUT


def test_missing_gguf_paths_raise_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(lightonocr.ENV_DRY_RUN, raising=False)
    monkeypatch.delenv(lightonocr.ENV_GGUF_MODEL_PATH, raising=False)
    monkeypatch.delenv(lightonocr.ENV_GGUF_MMPROJ_PATH, raising=False)

    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    with pytest.raises(ValueError) as excinfo:
        lightonocr.ocr_image(image)

    message = str(excinfo.value)
    assert lightonocr.ENV_GGUF_MODEL_PATH in message
    assert lightonocr.ENV_GGUF_MMPROJ_PATH in message


def test_missing_llama_cpp_import_is_actionable(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import ragprep.ocr.llamacpp_runtime as llamacpp_runtime

    llamacpp_runtime._get_runtime_cached.cache_clear()

    model_path = tmp_path / "model.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"
    model_path.write_text("x", encoding="utf-8")
    mmproj_path.write_text("y", encoding="utf-8")

    monkeypatch.setenv(lightonocr.ENV_GGUF_MODEL_PATH, str(model_path))
    monkeypatch.setenv(lightonocr.ENV_GGUF_MMPROJ_PATH, str(mmproj_path))

    def fake_import_module(_name: str) -> object:
        raise ModuleNotFoundError("nope")

    monkeypatch.setattr(llamacpp_runtime.importlib, "import_module", fake_import_module)

    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    with pytest.raises(ImportError, match="llama-cpp-python"):
        lightonocr.ocr_image(image)


def test_gguf_env_paths_strip_quotes_and_angle_brackets(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    model_path = tmp_path / "model.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"
    model_path.write_text("x", encoding="utf-8")
    mmproj_path.write_text("y", encoding="utf-8")

    monkeypatch.delenv(lightonocr.ENV_DRY_RUN, raising=False)
    monkeypatch.setenv(lightonocr.ENV_GGUF_MODEL_PATH, f"\"{model_path}\"")
    monkeypatch.setenv(lightonocr.ENV_GGUF_MMPROJ_PATH, f"<{mmproj_path}>")

    settings = lightonocr.get_settings()
    assert settings.gguf_model_path == str(model_path)
    assert settings.gguf_mmproj_path == str(mmproj_path)


def test_gguf_env_paths_support_file_uri(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    model_path = tmp_path / "model.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"
    model_path.write_text("x", encoding="utf-8")
    mmproj_path.write_text("y", encoding="utf-8")

    monkeypatch.delenv(lightonocr.ENV_DRY_RUN, raising=False)
    monkeypatch.setenv(lightonocr.ENV_GGUF_MODEL_PATH, model_path.resolve().as_uri())
    monkeypatch.setenv(lightonocr.ENV_GGUF_MMPROJ_PATH, mmproj_path.resolve().as_uri())

    settings = lightonocr.get_settings()
    assert settings.gguf_model_path == str(model_path)
    assert settings.gguf_mmproj_path == str(mmproj_path)


def test_llama_cpp_executes_with_mock_and_is_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import ragprep.ocr.llamacpp_runtime as llamacpp_runtime

    llamacpp_runtime._get_runtime_cached.cache_clear()

    created = []

    class FakeChatHandler:
        def __init__(self, clip_model_path: str) -> None:
            self.clip_model_path = clip_model_path

    class FakeLlama:
        def __init__(self, model_path: str, chat_handler: object, **_kwargs: object) -> None:
            self.model_path = model_path
            self.chat_handler = chat_handler
            self.calls = []
            created.append(self)

        def create_chat_completion(self, *, messages: object, max_tokens: int) -> dict[str, object]:
            self.calls.append({"messages": messages, "max_tokens": max_tokens})
            return {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr(llamacpp_runtime, "_import_llama_cpp", lambda: (FakeLlama, FakeChatHandler))

    model_path = tmp_path / "model.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"
    model_path.write_text("x", encoding="utf-8")
    mmproj_path.write_text("y", encoding="utf-8")

    monkeypatch.setenv(lightonocr.ENV_GGUF_MODEL_PATH, str(model_path))
    monkeypatch.setenv(lightonocr.ENV_GGUF_MMPROJ_PATH, str(mmproj_path))
    monkeypatch.setenv(lightonocr.ENV_MAX_NEW_TOKENS, "7")

    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    assert lightonocr.ocr_image(image) == "OK"
    assert lightonocr.ocr_image(image) == "OK"

    assert len(created) == 1
    llama = created[0]
    assert llama.model_path == str(model_path)
    assert getattr(llama.chat_handler, "clip_model_path", None) == str(mmproj_path)
    assert llama.calls[0]["max_tokens"] == 7
