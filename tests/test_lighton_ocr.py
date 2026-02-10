from __future__ import annotations

import pytest

from ragprep.config import get_settings
from ragprep.ocr import lighton_ocr


class _StubResponse:
    def __init__(self, *, status_code: int, payload: object, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        return self._payload


class _StubClient:
    def __init__(self, *, response: _StubResponse, capture: dict[str, object]) -> None:
        self._response = response
        self._capture = capture

    def __enter__(self) -> _StubClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        _ = exc_type, exc, tb
        return None

    def post(self, url: str, json: dict[str, object]) -> _StubResponse:
        self._capture["url"] = url
        self._capture["json"] = json
        return self._response


def test_lighton_ocr_posts_openai_compatible_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    capture: dict[str, object] = {}

    monkeypatch.setattr(
        lighton_ocr,
        "ensure_server_base_url",
        lambda _settings: "http://127.0.0.1:8080",
    )
    monkeypatch.setattr(
        lighton_ocr.httpx,
        "Client",
        lambda **_kwargs: _StubClient(
            response=_StubResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "OCR TEXT"}}]},
            ),
            capture=capture,
        ),
    )

    result = lighton_ocr.ocr_image_base64("aGVsbG8=", settings=settings)
    assert result == "OCR TEXT"

    payload = capture["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "lighton-ocr"
    assert "messages" in payload
    assert capture["url"] == "http://127.0.0.1:8080/v1/chat/completions"

    messages = payload.get("messages")
    assert isinstance(messages, list) and messages
    first = messages[0]
    assert isinstance(first, dict)
    content = first.get("content")
    assert isinstance(content, list) and content
    first_item = content[0]
    assert isinstance(first_item, dict)
    assert first_item.get("type") == "text"
    text_prompt = str(first_item.get("text") or "")
    assert "return only extracted content" in text_prompt.lower()
    assert "do not add explanations" in text_prompt.lower()


def test_lighton_ocr_reads_list_content(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(
        lighton_ocr,
        "ensure_server_base_url",
        lambda _settings: "http://127.0.0.1:8080",
    )
    monkeypatch.setattr(
        lighton_ocr.httpx,
        "Client",
        lambda **_kwargs: _StubClient(
            response=_StubResponse(
                status_code=200,
                payload={
                    "choices": [
                        {"message": {"content": [{"type": "text", "text": "A"}, {"text": "B"}]}}
                    ]
                },
            ),
            capture={},
        ),
    )

    result = lighton_ocr.ocr_image_base64("aGVsbG8=", settings=settings)
    assert result == "AB"


def test_lighton_ocr_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(
        lighton_ocr,
        "ensure_server_base_url",
        lambda _settings: "http://127.0.0.1:8080",
    )
    monkeypatch.setattr(
        lighton_ocr.httpx,
        "Client",
        lambda **_kwargs: _StubClient(
            response=_StubResponse(status_code=500, payload={}, text="boom"),
            capture={},
        ),
    )

    with pytest.raises(RuntimeError, match="HTTP 500"):
        _ = lighton_ocr.ocr_image_base64("aGVsbG8=", settings=settings)


def test_lighton_ocr_rejects_invalid_base64() -> None:
    settings = get_settings()
    with pytest.raises(ValueError, match="not valid base64"):
        _ = lighton_ocr.ocr_image_base64("not base64 !!!", settings=settings)


def test_lighton_ocr_rejects_base64_with_trailing_garbage() -> None:
    settings = get_settings()
    with pytest.raises(ValueError, match="not valid base64"):
        _ = lighton_ocr.ocr_image_base64("aGVsbG8=@@@@", settings=settings)
