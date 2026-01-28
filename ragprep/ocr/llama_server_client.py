from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from .llama_server_payload import build_ocr_chat_payload, extract_ocr_text

_TIMEOUT_ENV_NAME = "LIGHTONOCR_REQUEST_TIMEOUT_SECONDS"

logger = logging.getLogger(__name__)


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


def _format_timeout_seconds(seconds: float) -> str:
    if float(seconds).is_integer():
        return str(int(seconds))
    return str(seconds)


def _build_timeout(seconds: float) -> httpx.Timeout:
    return httpx.Timeout(connect=seconds, read=seconds, write=seconds, pool=seconds)


def ocr_image_base64(
    *,
    image_base64: str,
    prompt: str,
    settings: LlamaServerClientSettings,
    transport: httpx.BaseTransport | None = None,
) -> str:
    from ragprep.diagnostics import (
        diagnostics_dir,
        record_last_error,
        record_last_llama_request,
        sha256_text,
        summarize_base64,
    )

    url = _build_request_url(settings.base_url)
    request_id = uuid.uuid4().hex
    image_meta = summarize_base64(image_base64)
    request_meta: dict[str, Any] = {
        "request_id": request_id,
        "pid": os.getpid(),
        "url": url,
        "model": settings.model,
        "timeout_seconds": settings.timeout_seconds,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "prompt_len": len(prompt),
        "prompt_sha256": sha256_text(prompt),
        **image_meta,
        "diagnostics_dir": str(diagnostics_dir()),
    }
    diag_dir = request_meta["diagnostics_dir"]

    record_last_llama_request(request_meta)

    try:
        payload: dict[str, Any] = build_ocr_chat_payload(
            prompt=prompt,
            image_base64=image_base64,
            model=settings.model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
    except Exception as exc:  # noqa: BLE001
        record_last_error({**request_meta, "stage": "build_payload", "error": str(exc)})
        raise

    try:
        timeout = _build_timeout(settings.timeout_seconds)
        with httpx.Client(timeout=timeout, transport=transport) as client:
            response = client.post(url, json=payload)
    except httpx.ReadTimeout as exc:
        timeout_seconds = _format_timeout_seconds(settings.timeout_seconds)
        record_last_error(
            {
                **request_meta,
                "stage": "request_timeout",
                "timeout_formatted_seconds": timeout_seconds,
                "error": str(exc),
            }
        )
        message = (
            "llama-server request timed out "
            f"({url}; timeout={timeout_seconds}s). "
            f"Increase {_TIMEOUT_ENV_NAME} or ensure llama-server is running "
            f"(request_id={request_id}, "
            f"diagnostics_dir={diag_dir})."
        )
        raise RuntimeError(message) from exc
    except httpx.RequestError as exc:
        record_last_error({**request_meta, "stage": "request_error", "error": str(exc)})
        raise RuntimeError(
            f"llama-server request failed ({url}; request_id={request_id}, "
            f"diagnostics_dir={diag_dir}): {exc}"
        ) from exc

    if response.status_code != 200:
        body = (response.text or "").strip()
        detail = f" {body}" if body else ""
        record_last_error(
            {
                **request_meta,
                "stage": "non_200",
                "status_code": int(response.status_code),
                "response_body_len": len(body),
                "error": f"HTTP {response.status_code}",
            }
        )
        raise RuntimeError(
            f"llama-server request failed ({url}; request_id={request_id}): "
            f"HTTP {response.status_code}{detail} "
            f"(diagnostics_dir={diag_dir})"
        )

    try:
        data = response.json()
    except ValueError as exc:
        record_last_error({**request_meta, "stage": "invalid_json", "error": str(exc)})
        raise RuntimeError(
            f"llama-server response is not valid JSON "
            f"({url}; request_id={request_id}, "
            f"diagnostics_dir={diag_dir})"
        ) from exc

    return extract_ocr_text(data)
