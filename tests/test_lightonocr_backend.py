from __future__ import annotations

import pytest

from ragprep.ocr import lightonocr


def test_lightonocr_uses_llama_server_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTONOCR_BACKEND", "llama-server")
    monkeypatch.setenv("LIGHTONOCR_LLAMA_SERVER_URL", "http://server")
    monkeypatch.setenv("LIGHTONOCR_MODEL", "vision-model")
    monkeypatch.delenv("LIGHTONOCR_DRY_RUN", raising=False)

    def _fake_server(**_kwargs: object) -> str:
        return "SERVER"

    monkeypatch.setattr(lightonocr, "ocr_image_base64_llama_server", _fake_server)

    assert lightonocr.ocr_image_base64("aGVsbG8=") == "SERVER"


def test_lightonocr_llama_server_requires_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTONOCR_BACKEND", "llama-server")
    monkeypatch.setenv("LIGHTONOCR_LLAMA_SERVER_URL", "http://server")
    monkeypatch.delenv("LIGHTONOCR_MODEL", raising=False)
    monkeypatch.delenv("LIGHTONOCR_DRY_RUN", raising=False)

    with pytest.raises(ValueError, match="LIGHTONOCR_MODEL"):
        lightonocr.ocr_image_base64("aGVsbG8=")
