from __future__ import annotations

import base64
import json

import httpx
import pytest

from ragprep.config import get_settings
from ragprep.layout import glm_doclayout


def test_glm_doclayout_sends_openai_chat_completions_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        content = {
            "schema_version": "v1",
            "elements": [
                {"page_index": 0, "bbox": [1, 2, 3, 4], "label": "text", "score": 0.9},
            ],
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content)}}]},
        )

    monkeypatch.setattr("ragprep.layout.glm_doclayout._post_chat_completions", _fake_post)

    image_b64 = base64.b64encode(b"not-a-real-png").decode("ascii")
    result = glm_doclayout.analyze_layout_image_base64(image_b64, settings=settings)

    assert result["schema_version"] == "v1"
    assert isinstance(result["elements"], list)
    assert result["elements"][0]["label"] == "text"

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
    assert content[1]["text"] == glm_doclayout.DEFAULT_LAYOUT_ANALYSIS_PROMPT

    assert captured["timeout_seconds"] == 9


def test_glm_doclayout_requires_server_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODE", "transformers")
    settings = get_settings()
    image_b64 = base64.b64encode(b"x").decode("ascii")
    with pytest.raises(RuntimeError, match="requires RAGPREP_GLM_OCR_MODE=server"):
        glm_doclayout.analyze_layout_image_base64(image_b64, settings=settings)


def test_glm_doclayout_raises_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODE", "server")
    settings = get_settings()

    def _fake_post(**_kwargs: object) -> httpx.Response:
        return httpx.Response(500, content=b"oops")

    monkeypatch.setattr("ragprep.layout.glm_doclayout._post_chat_completions", _fake_post)

    image_b64 = base64.b64encode(b"x").decode("ascii")
    with pytest.raises(RuntimeError, match="returned 500"):
        glm_doclayout.analyze_layout_image_base64(image_b64, settings=settings)


def test_glm_doclayout_rejects_invalid_base64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODE", "server")
    settings = get_settings()
    with pytest.raises(ValueError, match="not valid base64"):
        glm_doclayout.analyze_layout_image_base64("not base64 !!!", settings=settings)


def test_glm_doclayout_rejects_non_json_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODE", "server")
    settings = get_settings()

    def _fake_post(**_kwargs: object) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not-json"}}]})

    monkeypatch.setattr("ragprep.layout.glm_doclayout._post_chat_completions", _fake_post)

    image_b64 = base64.b64encode(b"x").decode("ascii")
    with pytest.raises(RuntimeError, match="does not contain a JSON object"):
        glm_doclayout.analyze_layout_image_base64(image_b64, settings=settings)


def test_glm_doclayout_parses_fenced_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODE", "server")
    settings = get_settings()

    fenced = """```json
{"schema_version":"v1","elements":[{"page_index":0,"bbox":[1,2,3,4],"label":"text","score":0.9}]}
```"""

    def _fake_post(**_kwargs: object) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": fenced}}]})

    monkeypatch.setattr("ragprep.layout.glm_doclayout._post_chat_completions", _fake_post)

    image_b64 = base64.b64encode(b"x").decode("ascii")
    result = glm_doclayout.analyze_layout_image_base64(image_b64, settings=settings)
    assert result["schema_version"] == "v1"
    elements = result["elements"]
    assert isinstance(elements, list)
    assert isinstance(elements[0], dict)
    assert elements[0]["label"] == "text"


def test_glm_doclayout_extracts_json_from_surrounding_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_GLM_OCR_MODE", "server")
    settings = get_settings()

    content = (
        "Here is the result:\n"
        '{"schema_version":"v1","elements":[{"page_index":0,"bbox":[1,2,3,4],"label":"text"}]}\n'
        "done"
    )

    def _fake_post(**_kwargs: object) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("ragprep.layout.glm_doclayout._post_chat_completions", _fake_post)

    image_b64 = base64.b64encode(b"x").decode("ascii")
    result = glm_doclayout.analyze_layout_image_base64(image_b64, settings=settings)
    assert result["schema_version"] == "v1"
    elements = result["elements"]
    assert isinstance(elements, list)
    assert isinstance(elements[0], dict)
    assert elements[0]["page_index"] == 0
