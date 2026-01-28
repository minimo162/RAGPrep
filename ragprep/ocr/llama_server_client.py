from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from .llama_server_payload import build_ocr_chat_payload, extract_ocr_text

_TIMEOUT_ENV_NAME = "LIGHTONOCR_REQUEST_TIMEOUT_SECONDS"
_RETRY_ENV_NAME = "LIGHTONOCR_REQUEST_RETRIES"
_RETRY_BACKOFF_ENV_NAME = "LIGHTONOCR_RETRY_BACKOFF_BASE_SECONDS"

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

    def _get_non_negative_int_env(name: str, *, default: int, max_value: int | None = None) -> int:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = int(raw.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an int, got: {raw!r}") from exc
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got: {value}")
        if max_value is not None:
            value = min(value, max_value)
        return value

    def _get_non_negative_float_env(
        name: str, *, default: float, max_value: float | None = None
    ) -> float:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = float(raw.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be a float, got: {raw!r}") from exc
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got: {value}")
        if max_value is not None:
            value = min(value, max_value)
        return value

    request_retries = _get_non_negative_int_env(_RETRY_ENV_NAME, default=1, max_value=2)
    retry_backoff_base_seconds = _get_non_negative_float_env(
        _RETRY_BACKOFF_ENV_NAME, default=0.2, max_value=2.0
    )

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
        "request_retries": request_retries,
        "retry_backoff_base_seconds": retry_backoff_base_seconds,
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

    timeout = _build_timeout(settings.timeout_seconds)
    response: httpx.Response | None = None
    with httpx.Client(timeout=timeout, transport=transport) as client:
        for attempt in range(request_retries + 1):
            try:
                response = client.post(url, json=payload)
                break
            except httpx.ReadTimeout as exc:
                timeout_seconds = _format_timeout_seconds(settings.timeout_seconds)
                record_last_error(
                    {
                        **request_meta,
                        "stage": "request_timeout",
                        "attempt": attempt + 1,
                        "timeout_formatted_seconds": timeout_seconds,
                        "error": str(exc),
                    }
                )
                if attempt >= request_retries:
                    message = (
                        "llama-server request timed out "
                        f"({url}; timeout={timeout_seconds}s). "
                        f"Increase {_TIMEOUT_ENV_NAME} or ensure llama-server is running "
                        f"(request_id={request_id}, diagnostics_dir={diag_dir})."
                    )
                    raise RuntimeError(message) from exc

                if transport is None and retry_backoff_base_seconds > 0:
                    backoff_seconds = min(2.0, retry_backoff_base_seconds * (2**attempt))
                    time.sleep(backoff_seconds)
            except httpx.RequestError as exc:
                record_last_error(
                    {
                        **request_meta,
                        "stage": "request_error",
                        "attempt": attempt + 1,
                        "error": str(exc),
                    }
                )
                if attempt >= request_retries:
                    raise RuntimeError(
                        f"llama-server request failed ({url}; request_id={request_id}, "
                        f"diagnostics_dir={diag_dir}): {exc}"
                    ) from exc

                if transport is None and retry_backoff_base_seconds > 0:
                    backoff_seconds = min(2.0, retry_backoff_base_seconds * (2**attempt))
                    time.sleep(backoff_seconds)

    if response is None:
        raise RuntimeError(
            "llama-server request failed "
            f"({url}; request_id={request_id}, diagnostics_dir={diag_dir})"
        )

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
