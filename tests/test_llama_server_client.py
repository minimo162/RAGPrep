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


def test_llama_server_client_error_response_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"message": "bad request"}})

    transport = httpx.MockTransport(handler)
    settings = LlamaServerClientSettings(
        base_url="http://server",
        model="vision-model",
        timeout_seconds=10,
    )

    with pytest.raises(RuntimeError, match="bad request"):
        ocr_image_base64(
            image_base64="aGVsbG8=",
            prompt="Extract all text from this image and return it as Markdown.",
            settings=settings,
            transport=transport,
        )


def test_llama_server_client_retries_read_timeout_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    monkeypatch.setenv("LIGHTONOCR_REQUEST_RETRIES", "1")
    monkeypatch.setenv("LIGHTONOCR_RETRY_BACKOFF_BASE_SECONDS", "1.0")

    transport = httpx.MockTransport(handler)
    settings = LlamaServerClientSettings(
        base_url="http://server",
        model="vision-model",
        timeout_seconds=10,
    )

    result = ocr_image_base64(
        image_base64="aGVsbG8=",
        prompt="Extract all text from this image and return it as Markdown.",
        settings=settings,
        transport=transport,
    )
    assert result == "OK"
    assert len(calls) == 2


def test_llama_server_client_retries_request_error_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("connection failed", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    monkeypatch.setenv("LIGHTONOCR_REQUEST_RETRIES", "1")
    monkeypatch.setenv("LIGHTONOCR_RETRY_BACKOFF_BASE_SECONDS", "1.0")

    transport = httpx.MockTransport(handler)
    settings = LlamaServerClientSettings(
        base_url="http://server",
        model="vision-model",
        timeout_seconds=10,
    )

    result = ocr_image_base64(
        image_base64="aGVsbG8=",
        prompt="Extract all text from this image and return it as Markdown.",
        settings=settings,
        transport=transport,
    )
    assert result == "OK"
    assert len(calls) == 2


def test_llama_server_client_clamps_request_retries_to_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("connection failed", request=request)

    monkeypatch.setenv("LIGHTONOCR_REQUEST_RETRIES", "10")

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

    assert len(calls) == 3
