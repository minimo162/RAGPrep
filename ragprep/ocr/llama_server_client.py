from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .llama_server_payload import build_ocr_chat_payload, extract_ocr_text


@dataclass(frozen=True)
class LlamaServerClientSettings:
    base_url: str
    model: str
    timeout_seconds: float
    max_tokens: int | None = None
    temperature: float | None = None


def _normalize_base_url(base_url: str) -> str:
    if base_url is None:
        raise ValueError("llama-server base_url is required")
    trimmed = base_url.strip().rstrip("/")
    if not trimmed:
        raise ValueError("llama-server base_url is required")
    return trimmed


def _build_request_url(base_url: str) -> str:
    return f"{_normalize_base_url(base_url)}/v1/chat/completions"


def ocr_image_base64(
    *,
    image_base64: str,
    prompt: str,
    settings: LlamaServerClientSettings,
    transport: httpx.BaseTransport | None = None,
) -> str:
    url = _build_request_url(settings.base_url)
    payload: dict[str, Any] = build_ocr_chat_payload(
        prompt=prompt,
        image_base64=image_base64,
        model=settings.model,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )

    try:
        with httpx.Client(timeout=settings.timeout_seconds, transport=transport) as client:
            response = client.post(url, json=payload)
    except httpx.RequestError as exc:
        raise RuntimeError(f"llama-server request failed ({url}): {exc}") from exc

    if response.status_code != 200:
        body = (response.text or "").strip()
        detail = f" {body}" if body else ""
        raise RuntimeError(
            f"llama-server request failed ({url}): HTTP {response.status_code}{detail}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"llama-server response is not valid JSON ({url})") from exc

    return extract_ocr_text(data)
