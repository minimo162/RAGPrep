from __future__ import annotations

import pytest

from ragprep import config


def test_default_ocr_backend_is_lighton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    monkeypatch.delenv("RAGPREP_OCR_BACKEND", raising=False)
    settings = config.get_settings()
    assert settings.ocr_backend == "lighton-ocr"
    assert settings.pdf_backend == "lighton-ocr"


def test_ocr_backend_accepts_lighton_and_glm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_PDF_BACKEND", "LIGHTON-OCR")
    settings = config.get_settings()
    assert settings.ocr_backend == "lighton-ocr"
    assert settings.pdf_backend == "lighton-ocr"

    monkeypatch.setenv("RAGPREP_OCR_BACKEND", "glm-ocr")
    settings = config.get_settings()
    assert settings.ocr_backend == "glm-ocr"
    assert settings.pdf_backend == "glm-ocr"


def test_ocr_backend_rejects_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_OCR_BACKEND", "invalid")
    with pytest.raises(ValueError, match="RAGPREP_PDF_BACKEND"):
        config.get_settings()


def test_lighton_env_values_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_BASE_URL", "  http://localhost:8080  ")
    monkeypatch.setenv("RAGPREP_LIGHTON_MODEL", "  noctrex/LightOnOCR-2-1B-GGUF  ")
    monkeypatch.setenv("RAGPREP_LIGHTON_API_KEY", "  secret ")
    monkeypatch.setenv("RAGPREP_LIGHTON_MAX_TOKENS", "  123  ")
    monkeypatch.setenv("RAGPREP_LIGHTON_TIMEOUT_SECONDS", "  9  ")
    settings = config.get_settings()
    assert settings.lighton_base_url == "http://localhost:8080"
    assert settings.lighton_model == "noctrex/LightOnOCR-2-1B-GGUF"
    assert settings.lighton_api_key == "secret"
    assert settings.lighton_max_tokens == 123
    assert settings.lighton_timeout_seconds == 9


def test_default_layout_mode_is_lighton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_LAYOUT_MODE", raising=False)
    settings = config.get_settings()
    assert settings.layout_mode == "lighton"


def test_layout_settings_default_to_lighton_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_LAYOUT_MODE", raising=False)
    monkeypatch.delenv("RAGPREP_LAYOUT_BASE_URL", raising=False)
    monkeypatch.delenv("RAGPREP_LAYOUT_MODEL", raising=False)
    monkeypatch.delenv("RAGPREP_LAYOUT_API_KEY", raising=False)
    monkeypatch.delenv("RAGPREP_LAYOUT_MAX_TOKENS", raising=False)
    monkeypatch.delenv("RAGPREP_LAYOUT_TIMEOUT_SECONDS", raising=False)

    monkeypatch.setenv("RAGPREP_LIGHTON_BASE_URL", "http://lighton:8080")
    monkeypatch.setenv("RAGPREP_LIGHTON_MODEL", "lighton-model")
    monkeypatch.setenv("RAGPREP_LIGHTON_API_KEY", "k")
    monkeypatch.setenv("RAGPREP_LIGHTON_MAX_TOKENS", "111")
    monkeypatch.setenv("RAGPREP_LIGHTON_TIMEOUT_SECONDS", "7")

    settings = config.get_settings()
    assert settings.layout_mode == "lighton"
    assert settings.layout_base_url == "http://lighton:8080"
    assert settings.layout_model == "lighton-model"
    assert settings.layout_api_key == "k"
    assert settings.layout_max_tokens == 111
    assert settings.layout_timeout_seconds == 7


def test_layout_settings_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_layout_mode_accepts_lighton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_LAYOUT_MODE", "lighton")
    settings = config.get_settings()
    assert settings.layout_mode == "lighton"
