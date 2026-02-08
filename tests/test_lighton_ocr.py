from __future__ import annotations

import base64
from io import BytesIO
from typing import cast

import httpx
import pytest
from PIL import Image

from ragprep import config
from ragprep.ocr import lighton_ocr


def _tiny_png_base64() -> str:
    image = Image.new("RGB", (2, 2), color=(255, 255, 255))
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_lighton_ocr_sends_openai_chat_completions_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGPREP_LIGHTON_BASE_URL", "http://localhost:8080/")
    monkeypatch.setenv("RAGPREP_LIGHTON_MODEL", "noctrex/LightOnOCR-2-1B-GGUF")
    monkeypatch.setenv("RAGPREP_LIGHTON_API_KEY", "secret")
    monkeypatch.setenv("RAGPREP_LIGHTON_MAX_TOKENS", "123")
    monkeypatch.setenv("RAGPREP_LIGHTON_TIMEOUT_SECONDS", "9")
    settings = config.get_settings()

    image_b64 = _tiny_png_base64()
    captured: dict[str, object] = {}

    class _Resp:
        status_code = 200

        @property
        def text(self) -> str:
            return "ok"

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"schema_version":"v1","elements":[{"page_index":0,"bbox":[0,0,10,10],'
                                '"label":"text","score":0.9}],"lines":[{"bbox":[0,0,10,10],"text":"abc"}]}'
                            )
                        }
                    }
                ]
            }

    def _fake_post(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: int,
    ) -> _Resp:
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        captured["timeout_seconds"] = timeout_seconds
        return _Resp()

    monkeypatch.setattr("ragprep.ocr.lighton_ocr._post_chat_completions", _fake_post)

    result = lighton_ocr.analyze_ocr_layout_image_base64(image_b64, settings=settings)

    assert captured["url"] == "http://localhost:8080/v1/chat/completions"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer secret"

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "noctrex/LightOnOCR-2-1B-GGUF"
    assert payload["max_tokens"] == 123
    messages = payload["messages"]
    assert isinstance(messages, list)
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"
    assert str(content[0]["image_url"]["url"]).startswith("data:image/png;base64,")
    assert result["schema_version"] == "v1"
    elements = cast(list[dict[str, object]], result["elements"])
    lines = cast(list[dict[str, object]], result["lines"])
    assert len(elements) == 1
    assert len(lines) == 1


def test_lighton_ocr_parses_fenced_json(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = config.get_settings()
    image_b64 = _tiny_png_base64()

    class _Resp:
        status_code = 200

        @property
        def text(self) -> str:
            return "ok"

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "```json\n"
                                '{"schema_version":"v1","elements":[{"page_index":0,"bbox":[0,0,10,10],'
                                '"label":"text"}],"lines":[{"bbox":[0,0,10,10],"text":"x"}]}\n'
                                "```"
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr("ragprep.ocr.lighton_ocr._post_chat_completions", lambda **kwargs: _Resp())

    result = lighton_ocr.analyze_ocr_layout_image_base64(image_b64, settings=settings)
    assert result["schema_version"] == "v1"
    elements = cast(list[dict[str, object]], result["elements"])
    lines = cast(list[dict[str, object]], result["lines"])
    assert elements[0]["label"] == "text"
    assert lines[0]["text"] == "x"


def test_lighton_ocr_raises_on_missing_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = config.get_settings()
    image_b64 = _tiny_png_base64()

    class _Resp:
        status_code = 200

        @property
        def text(self) -> str:
            return "ok"

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"schema_version":"v1","elements":[{"page_index":0,"bbox":[0,0,10,10],'
                                '"label":"text"}]}'
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr("ragprep.ocr.lighton_ocr._post_chat_completions", lambda **kwargs: _Resp())
    with pytest.raises(RuntimeError, match="lines list"):
        lighton_ocr.analyze_ocr_layout_image_base64(image_b64, settings=settings)


def test_lighton_ocr_timeout_maps_to_stable_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = config.get_settings()
    image_b64 = _tiny_png_base64()

    def _fake_post(**_kwargs: object) -> object:
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr("ragprep.ocr.lighton_ocr._post_chat_completions", _fake_post)
    with pytest.raises(RuntimeError, match="LightOn OCR request timed out"):
        lighton_ocr.analyze_ocr_layout_image_base64(image_b64, settings=settings)
