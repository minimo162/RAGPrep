from __future__ import annotations

import pytest

from ragprep import config


def test_default_ocr_backend_is_glm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    monkeypatch.delenv("RAGPREP_OCR_BACKEND", raising=False)
    settings = config.get_settings()
    assert settings.ocr_backend == "glm-ocr"
    assert settings.pdf_backend == "glm-ocr"


def test_ocr_backend_accepts_only_glm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_PDF_BACKEND", "GLM-OCR")
    settings = config.get_settings()
    assert settings.ocr_backend == "glm-ocr"
    assert settings.pdf_backend == "glm-ocr"

    monkeypatch.setenv("RAGPREP_OCR_BACKEND", "glm-ocr")
    settings = config.get_settings()
    assert settings.ocr_backend == "glm-ocr"
    assert settings.pdf_backend == "glm-ocr"


def test_ocr_backend_rejects_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_OCR_BACKEND", "invalid")
    with pytest.raises(ValueError, match="RAGPREP_PDF_BACKEND"):
        config.get_settings()


def test_ocr_backend_rejects_lighton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_PDF_BACKEND", "lighton-ocr")
    with pytest.raises(ValueError, match="RAGPREP_PDF_BACKEND"):
        config.get_settings()


def test_glm_ocr_mode_rejects_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODE", "server")
    with pytest.raises(ValueError, match="RAGPREP_GLM_OCR_MODE"):
        config.get_settings()


def test_default_layout_mode_is_local_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_LAYOUT_MODE", raising=False)
    settings = config.get_settings()
    assert settings.layout_mode == "local-fast"


def test_layout_settings_default_to_glm_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_LAYOUT_MODE", raising=False)
    monkeypatch.delenv("RAGPREP_LAYOUT_BASE_URL", raising=False)
    monkeypatch.delenv("RAGPREP_LAYOUT_MODEL", raising=False)
    monkeypatch.delenv("RAGPREP_LAYOUT_API_KEY", raising=False)
    monkeypatch.delenv("RAGPREP_LAYOUT_MAX_TOKENS", raising=False)
    monkeypatch.delenv("RAGPREP_LAYOUT_TIMEOUT_SECONDS", raising=False)

    monkeypatch.setenv("RAGPREP_GLM_OCR_BASE_URL", "http://glm-local:8080")
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODEL", "glm-local-model")
    monkeypatch.setenv("RAGPREP_GLM_OCR_API_KEY", "k")
    monkeypatch.setenv("RAGPREP_GLM_OCR_MAX_TOKENS", "111")
    monkeypatch.setenv("RAGPREP_GLM_OCR_TIMEOUT_SECONDS", "7")

    settings = config.get_settings()
    assert settings.layout_mode == "local-fast"
    assert settings.layout_base_url == "http://glm-local:8080"
    assert settings.layout_model == "glm-local-model"
    assert settings.layout_api_key == "k"
    assert settings.layout_max_tokens == 111
    assert settings.layout_timeout_seconds == 7


def test_layout_settings_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "local-paddle")
    monkeypatch.setenv("RAGPREP_LAYOUT_BASE_URL", "http://layout:8080")
    monkeypatch.setenv("RAGPREP_LAYOUT_MODEL", "layout-model")
    monkeypatch.setenv("RAGPREP_LAYOUT_API_KEY", "layout-key")
    monkeypatch.setenv("RAGPREP_LAYOUT_MAX_TOKENS", "222")
    monkeypatch.setenv("RAGPREP_LAYOUT_TIMEOUT_SECONDS", "9")

    settings = config.get_settings()
    assert settings.layout_mode == "local-paddle"
    assert settings.layout_base_url == "http://layout:8080"
    assert settings.layout_model == "layout-model"
    assert settings.layout_api_key == "layout-key"
    assert settings.layout_max_tokens == 222
    assert settings.layout_timeout_seconds == 9


def test_layout_mode_accepts_local_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "local-fast")
    settings = config.get_settings()
    assert settings.layout_mode == "local-fast"


def test_layout_mode_aliases_transformers_to_local_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "transformers")
    settings = config.get_settings()
    assert settings.layout_mode == "local-fast"


def test_layout_mode_accepts_local_paddle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "local-paddle")
    settings = config.get_settings()
    assert settings.layout_mode == "local-paddle"


def test_layout_mode_rejects_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "server")
    with pytest.raises(ValueError, match="RAGPREP_LAYOUT_MODE"):
        config.get_settings()


def test_layout_mode_rejects_lighton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "lighton")
    with pytest.raises(ValueError, match="RAGPREP_LAYOUT_MODE"):
        config.get_settings()
