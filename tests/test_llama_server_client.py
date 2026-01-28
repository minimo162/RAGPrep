from __future__ import annotations

import json

import httpx
import pytest

from ragprep.ocr.llama_server_client import LlamaServerClientSettings, ocr_image_base64


def test_llama_server_client_sends_payload_and_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://server/v1/chat/completions"
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "vision-model"
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    transport = httpx.MockTransport(handler)
    settings = LlamaServerClientSettings(
        base_url="http://server/",
        model="vision-model",
        timeout_seconds=10,
        max_tokens=32,
        temperature=0.2,
    )

    result = ocr_image_base64(
        image_base64="aGVsbG8=",
        prompt="Extract all text from this image and return it as Markdown.",
        settings=settings,
        transport=transport,
    )
    assert result == "OK"


def test_llama_server_client_non_200_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    settings = LlamaServerClientSettings(
        base_url="http://server",
        model="vision-model",
        timeout_seconds=10,
    )

    with pytest.raises(RuntimeError, match="HTTP 500"):
        ocr_image_base64(
            image_base64="aGVsbG8=",
            prompt="Extract all text from this image and return it as Markdown.",
            settings=settings,
            transport=transport,
        )


def test_llama_server_client_invalid_json_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{")

    transport = httpx.MockTransport(handler)
    settings = LlamaServerClientSettings(
        base_url="http://server",
        model="vision-model",
        timeout_seconds=10,
    )

    with pytest.raises(RuntimeError, match="not valid JSON"):
        ocr_image_base64(
            image_base64="aGVsbG8=",
            prompt="Extract all text from this image and return it as Markdown.",
            settings=settings,
            transport=transport,
        )


def test_llama_server_client_request_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    transport = httpx.MockTransport(handler)
    settings = LlamaServerClientSettings(
        base_url="http://server",
        model="vision-model",
        timeout_seconds=10,
    )

    with pytest.raises(RuntimeError, match="request failed"):
        ocr_image_base64(
            image_base64="aGVsbG8=",
            prompt="Extract all text from this image and return it as Markdown.",
            settings=settings,
            transport=transport,
        )


def test_llama_server_client_read_timeout_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    transport = httpx.MockTransport(handler)
    settings = LlamaServerClientSettings(
        base_url="http://server",
        model="vision-model",
        timeout_seconds=10,
    )

    with pytest.raises(RuntimeError) as exc_info:
        ocr_image_base64(
            image_base64="aGVsbG8=",
            prompt="Extract all text from this image and return it as Markdown.",
            settings=settings,
            transport=transport,
        )

    message = str(exc_info.value)
    assert "timed out" in message
    assert "timeout=10s" in message
    assert "LIGHTONOCR_REQUEST_TIMEOUT_SECONDS" in message
