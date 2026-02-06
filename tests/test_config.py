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


def test_pdf_backend_rejects_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_PDF_BACKEND", "invalid")
    with pytest.raises(ValueError, match="RAGPREP_PDF_BACKEND"):
        config.get_settings()


def test_glm_ocr_env_values_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    monkeypatch.delenv("RAGPREP_GLM_OCR_MODE", raising=False)
    monkeypatch.setenv("RAGPREP_GLM_OCR_BASE_URL", "  http://localhost:8080  ")
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODEL", "  zai-org/GLM-OCR  ")
    monkeypatch.setenv("RAGPREP_GLM_OCR_API_KEY", "  secret ")
    monkeypatch.setenv("RAGPREP_GLM_OCR_MAX_TOKENS", "  123  ")
    monkeypatch.setenv("RAGPREP_GLM_OCR_TIMEOUT_SECONDS", "  9  ")
    settings = config.get_settings()
    assert settings.glm_ocr_base_url == "http://localhost:8080"
    assert settings.glm_ocr_model == "zai-org/GLM-OCR"
    assert settings.glm_ocr_mode == "transformers"
    assert settings.glm_ocr_api_key == "secret"
    assert settings.glm_ocr_max_tokens == 123
    assert settings.glm_ocr_timeout_seconds == 9


def test_default_glm_ocr_mode_is_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_GLM_OCR_MODE", raising=False)
    settings = config.get_settings()
    assert settings.glm_ocr_mode == "transformers"


def test_layout_mode_defaults_to_local_paddle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_LAYOUT_MODE", raising=False)
    settings = config.get_settings()
    assert settings.layout_mode == "local-paddle"


def test_layout_settings_default_to_glm_ocr_values_when_layout_not_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAGPREP_LAYOUT_MODE", raising=False)
    monkeypatch.delenv("RAGPREP_LAYOUT_BASE_URL", raising=False)
    monkeypatch.delenv("RAGPREP_LAYOUT_MODEL", raising=False)
    monkeypatch.delenv("RAGPREP_LAYOUT_API_KEY", raising=False)
    monkeypatch.delenv("RAGPREP_LAYOUT_MAX_TOKENS", raising=False)
    monkeypatch.delenv("RAGPREP_LAYOUT_TIMEOUT_SECONDS", raising=False)

    monkeypatch.setenv("RAGPREP_GLM_OCR_BASE_URL", "http://example:8080")
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODEL", "model-x")
    monkeypatch.setenv("RAGPREP_GLM_OCR_API_KEY", "k")
    monkeypatch.setenv("RAGPREP_GLM_OCR_MAX_TOKENS", "111")
    monkeypatch.setenv("RAGPREP_GLM_OCR_TIMEOUT_SECONDS", "7")

    settings = config.get_settings()
    assert settings.layout_base_url == "http://example:8080"
    assert settings.layout_model == "model-x"
    assert settings.layout_api_key == "k"
    assert settings.layout_max_tokens == 111
    assert settings.layout_timeout_seconds == 7


def test_layout_settings_override_glm_ocr_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODE", "transformers")
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "server")
    monkeypatch.setenv("RAGPREP_LAYOUT_BASE_URL", "http://layout:8080")
    monkeypatch.setenv("RAGPREP_LAYOUT_MODEL", "layout-model")
    monkeypatch.setenv("RAGPREP_LAYOUT_API_KEY", "layout-key")
    monkeypatch.setenv("RAGPREP_LAYOUT_MAX_TOKENS", "222")
    monkeypatch.setenv("RAGPREP_LAYOUT_TIMEOUT_SECONDS", "9")

    settings = config.get_settings()
    assert settings.layout_mode == "server"
    assert settings.layout_base_url == "http://layout:8080"
    assert settings.layout_model == "layout-model"
    assert settings.layout_api_key == "layout-key"
    assert settings.layout_max_tokens == 222
    assert settings.layout_timeout_seconds == 9


def test_layout_mode_accepts_local_paddle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "local-paddle")
    settings = config.get_settings()
    assert settings.layout_mode == "local-paddle"
