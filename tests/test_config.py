from __future__ import annotations

import pytest

from ragprep import config


def test_default_pdf_backend_is_glm_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    settings = config.get_settings()
    assert settings.pdf_backend == "glm-ocr"


def test_pdf_backend_accepts_glm_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_PDF_BACKEND", "GLM-OCR")
    settings = config.get_settings()
    assert settings.pdf_backend == "glm-ocr"


def test_pdf_backend_accepts_lightonocr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_PDF_BACKEND", "LightOnOCR")
    settings = config.get_settings()
    assert settings.pdf_backend == "lightonocr"


def test_pdf_backend_rejects_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_PDF_BACKEND", "invalid")
    with pytest.raises(ValueError, match="RAGPREP_PDF_BACKEND"):
        config.get_settings()


def test_lightonocr_env_values_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    monkeypatch.setenv("LIGHTONOCR_GGUF_MODEL_PATH", "  model.gguf  ")
    monkeypatch.setenv("LIGHTONOCR_GGUF_MMPROJ_PATH", "  mmproj.gguf ")
    monkeypatch.setenv("LIGHTONOCR_LLAVA_CLI_PATH", "  cli.exe ")
    settings = config.get_settings()
    assert settings.lightonocr_model_path == "model.gguf"
    assert settings.lightonocr_mmproj_path == "mmproj.gguf"
    assert settings.lightonocr_llava_cli_path == "cli.exe"


def test_glm_ocr_env_values_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    monkeypatch.setenv("RAGPREP_GLM_OCR_BASE_URL", "  http://localhost:8080  ")
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODEL", "  zai-org/GLM-OCR  ")
    monkeypatch.setenv("RAGPREP_GLM_OCR_API_KEY", "  secret ")
    monkeypatch.setenv("RAGPREP_GLM_OCR_MAX_TOKENS", "  123  ")
    monkeypatch.setenv("RAGPREP_GLM_OCR_TIMEOUT_SECONDS", "  9  ")
    settings = config.get_settings()
    assert settings.glm_ocr_base_url == "http://localhost:8080"
    assert settings.glm_ocr_model == "zai-org/GLM-OCR"
    assert settings.glm_ocr_api_key == "secret"
    assert settings.glm_ocr_max_tokens == 123
    assert settings.glm_ocr_timeout_seconds == 9
