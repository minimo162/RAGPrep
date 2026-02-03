from __future__ import annotations

import base64

import httpx
import pytest

from ragprep.config import get_settings
from ragprep.ocr import glm_ocr


def test_glm_ocr_sends_openai_chat_completions_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODE", "server")
    monkeypatch.setenv("RAGPREP_GLM_OCR_BASE_URL", "http://localhost:8080/")
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODEL", "zai-org/GLM-OCR")
    monkeypatch.setenv("RAGPREP_GLM_OCR_API_KEY", "secret")
    monkeypatch.setenv("RAGPREP_GLM_OCR_MAX_TOKENS", "123")
    monkeypatch.setenv("RAGPREP_GLM_OCR_TIMEOUT_SECONDS", "9")
    settings = get_settings()

    captured: dict[str, object] = {}

    def _fake_post(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: int,
    ) -> httpx.Response:
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        captured["timeout_seconds"] = timeout_seconds
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    monkeypatch.setattr("ragprep.ocr.glm_ocr._post_chat_completions", _fake_post)

    image_b64 = base64.b64encode(b"not-a-real-png").decode("ascii")
    result = glm_ocr.ocr_image_base64(image_b64, settings=settings)
    assert result == "OK"

    assert captured["url"] == "http://localhost:8080/v1/chat/completions"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer secret"

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "zai-org/GLM-OCR"
    assert payload["max_tokens"] == 123

    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["type"] == "text"
    assert content[1]["text"] == glm_ocr.DEFAULT_TEXT_RECOGNITION_PROMPT

    assert captured["timeout_seconds"] == 9


def test_glm_ocr_raises_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODE", "server")
    settings = get_settings()

    def _fake_post(**_kwargs: object) -> httpx.Response:
        return httpx.Response(500, content=b"oops")

    monkeypatch.setattr("ragprep.ocr.glm_ocr._post_chat_completions", _fake_post)

    image_b64 = base64.b64encode(b"x").decode("ascii")
    with pytest.raises(RuntimeError, match="returned 500"):
        glm_ocr.ocr_image_base64(image_b64, settings=settings)


def test_glm_ocr_raises_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODE", "server")
    settings = get_settings()

    def _fake_post(**_kwargs: object) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    monkeypatch.setattr("ragprep.ocr.glm_ocr._post_chat_completions", _fake_post)

    image_b64 = base64.b64encode(b"x").decode("ascii")
    with pytest.raises(RuntimeError, match="invalid JSON"):
        glm_ocr.ocr_image_base64(image_b64, settings=settings)


def test_glm_ocr_raises_on_missing_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODE", "server")
    settings = get_settings()

    def _fake_post(**_kwargs: object) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {}}]})

    monkeypatch.setattr("ragprep.ocr.glm_ocr._post_chat_completions", _fake_post)

    image_b64 = base64.b64encode(b"x").decode("ascii")
    with pytest.raises(RuntimeError, match="message\\.content"):
        glm_ocr.ocr_image_base64(image_b64, settings=settings)


def test_glm_ocr_rejects_invalid_base64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODE", "server")
    settings = get_settings()
    with pytest.raises(ValueError, match="not valid base64"):
        glm_ocr.ocr_image_base64("not base64 !!!", settings=settings)
