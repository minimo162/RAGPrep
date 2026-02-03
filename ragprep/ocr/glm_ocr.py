from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass

import httpx

from ragprep.config import Settings

DEFAULT_TEXT_RECOGNITION_PROMPT = "Text Recognition:"


@dataclass(frozen=True)
class _ChatCompletionResult:
    content: str


def _normalize_base_url(base_url: str) -> str:
    return base_url.strip().rstrip("/")


def _strip_data_url_prefix(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    if raw.startswith("data:"):
        comma_index = raw.find(",")
        if comma_index != -1:
            return raw[comma_index + 1 :].strip()
    return raw


def _validate_base64_payload(payload: str) -> None:
    try:
        decoded = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc
    if not decoded:
        raise ValueError("image_base64 is empty")


def _post_chat_completions(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, object],
    timeout_seconds: int,
) -> httpx.Response:
    timeout = httpx.Timeout(timeout_seconds)
    with httpx.Client(timeout=timeout) as client:
        return client.post(url, headers=headers, json=payload)


def _parse_chat_completions_response(response: httpx.Response) -> _ChatCompletionResult:
    if response.status_code != 200:
        body = (response.text or "").strip()
        if len(body) > 800:
            body = body[:800] + "…"
        raise RuntimeError(f"GLM-OCR server returned {response.status_code}: {body}")

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("GLM-OCR server returned invalid JSON.") from exc

    if not isinstance(data, dict):
        raise RuntimeError("GLM-OCR server returned invalid JSON shape (expected object).")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("GLM-OCR response missing choices.")

    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("GLM-OCR response choices[0] is invalid.")

    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("GLM-OCR response missing message.")

    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("GLM-OCR response message.content is missing or not a string.")

    return _ChatCompletionResult(content=content)


def ocr_image_base64(image_base64: str, *, settings: Settings) -> str:
    """
    Call a locally running GLM-OCR server (OpenAI-compatible) and return extracted Markdown/text.

    This expects an OpenAI-compatible endpoint at:
      POST {RAGPREP_GLM_OCR_BASE_URL}/v1/chat/completions
    """

    if not image_base64:
        raise ValueError("image_base64 is empty")

    payload = _strip_data_url_prefix(image_base64)
    if not payload:
        raise ValueError("image_base64 is empty")
    _validate_base64_payload(payload)

    url = f"{_normalize_base_url(settings.glm_ocr_base_url)}/v1/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.glm_ocr_api_key is not None:
        headers["Authorization"] = f"Bearer {settings.glm_ocr_api_key}"

    data_url = f"data:image/png;base64,{payload}"
    request_payload: dict[str, object] = {
        "model": settings.glm_ocr_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": DEFAULT_TEXT_RECOGNITION_PROMPT},
                ],
            }
        ],
        "max_tokens": settings.glm_ocr_max_tokens,
    }

    try:
        response = _post_chat_completions(
            url=url,
            headers=headers,
            payload=request_payload,
            timeout_seconds=settings.glm_ocr_timeout_seconds,
        )
    except httpx.TimeoutException as exc:
        raise RuntimeError("GLM-OCR request timed out.") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Failed to reach GLM-OCR server: {exc}") from exc

    return _parse_chat_completions_response(response).content

