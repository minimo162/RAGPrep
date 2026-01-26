from __future__ import annotations

import pytest

from ragprep import config


def test_default_pdf_backend_is_lightonocr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGPREP_PDF_BACKEND", raising=False)
    settings = config.get_settings()
    assert settings.pdf_backend == "lightonocr"


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
