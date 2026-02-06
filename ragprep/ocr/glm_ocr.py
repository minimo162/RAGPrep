from __future__ import annotations

import base64
import binascii
import os
import tempfile
from dataclasses import dataclass
from threading import Lock
from typing import Any

import httpx
from PIL import Image

from ragprep.config import Settings
from ragprep.model_cache import configure_model_cache

DEFAULT_TEXT_RECOGNITION_PROMPT = "Text Recognition:"


@dataclass(frozen=True)
class _ChatCompletionResult:
    content: str


class _GlmOcrMode:
    transformers = "transformers"
    server = "server"


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


def _decode_png_from_base64(image_base64: str) -> bytes:
    payload = _strip_data_url_prefix(image_base64)
    if not payload:
        raise ValueError("image_base64 is empty")
    _validate_base64_payload(payload)
    return base64.b64decode(payload, validate=False)


def _ocr_image_base64_via_server(image_base64: str, *, settings: Settings) -> str:
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

    base_url = _normalize_base_url(settings.glm_ocr_base_url)
    try:
        response = _post_chat_completions(
            url=url,
            headers=headers,
            payload=request_payload,
            timeout_seconds=settings.glm_ocr_timeout_seconds,
        )
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            "GLM-OCR request timed out. "
            f"base_url={base_url!r}. "
            "Ensure the GLM-OCR server is running and reachable."
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(
            "Failed to reach GLM-OCR server. "
            f"base_url={base_url!r}. "
            f"error={exc}. "
            "Ensure the GLM-OCR server is running and reachable."
        ) from exc

    return _parse_chat_completions_response(response).content


@dataclass(frozen=True)
class _TransformersClient:
    model_path: str
    processor: Any
    model: Any


_transformers_client_lock = Lock()
_transformers_client: _TransformersClient | None = None


def _load_transformers_client(model_path: str) -> _TransformersClient:
    global _transformers_client

    with _transformers_client_lock:
        if _transformers_client is not None and _transformers_client.model_path == model_path:
            return _transformers_client

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Transformers backend selected, but required packages are missing. "
                "Install dependencies (example): "
                "pip install git+https://github.com/huggingface/transformers.git torch"
            ) from exc

        processor_cls: Any = AutoProcessor
        model_cls: Any = AutoModelForImageTextToText

        try:
            try:
                processor = processor_cls.from_pretrained(model_path, trust_remote_code=True)
            except TypeError:
                processor = processor_cls.from_pretrained(model_path)
        except Exception as exc:  # noqa: BLE001
            try:
                from transformers import __version__ as transformers_version
            except Exception:
                transformers_version = "unknown"
            raise RuntimeError(
                "Failed to load GLM-OCR processor via Transformers. "
                f"model={model_path!r}, transformers={transformers_version}. "
                "Try upgrading Transformers, or set RAGPREP_GLM_OCR_MODE=server."
            ) from exc

        model_kwargs: dict[str, object] = {"pretrained_model_name_or_path": model_path}
        # Try to honor the upstream recommended args, but fall back if unsupported.
        for key, value in (("torch_dtype", "auto"), ("device_map", "auto")):
            model_kwargs[key] = value
        model_kwargs["trust_remote_code"] = True

        try:
            try:
                model = model_cls.from_pretrained(**model_kwargs)
            except TypeError:
                try:
                    model = model_cls.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                    )
                except TypeError:
                    model = model_cls.from_pretrained(model_path)
        except Exception as exc:  # noqa: BLE001
            try:
                from transformers import __version__ as transformers_version
            except Exception:
                transformers_version = "unknown"
            raise RuntimeError(
                "Failed to load GLM-OCR model via Transformers. "
                f"model={model_path!r}, transformers={transformers_version}. "
                "Try upgrading Transformers, or set RAGPREP_GLM_OCR_MODE=server."
            ) from exc

        # Ensure model is on some device when device_map isn't supported.
        try:
            _ = model.device
        except Exception:
            try:
                model = model.to(torch.device("cpu"))
            except Exception:
                pass

        _transformers_client = _TransformersClient(
            model_path=model_path,
            processor=processor,
            model=model,
        )
        return _transformers_client


def _ocr_image_base64_via_transformers(image_base64: str, *, settings: Settings) -> str:
    if not image_base64:
        raise ValueError("image_base64 is empty")

    configure_model_cache(settings)

    png_bytes = _decode_png_from_base64(image_base64)

    # The official GLM-OCR transformers example uses a file path in messages.
    # Writing to a temp file avoids relying on in-memory image object support.
    tmp_path = ""
    fd: int | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="ragprep-glm-ocr-", suffix=".png")
        with os.fdopen(fd, "wb") as f:
            f.write(png_bytes)
        fd = None

        with Image.open(tmp_path) as image:
            image.verify()

        client = _load_transformers_client(settings.glm_ocr_model)
        processor = client.processor
        model = client.model

        try:
            import torch
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Transformers backend selected, but torch is not installed. "
                "Install torch and retry."
            ) from exc

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": tmp_path},
                    {"type": "text", "text": DEFAULT_TEXT_RECOGNITION_PROMPT},
                ],
            }
        ]

        inputs: Any = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        try:
            inputs = inputs.to(model.device)
        except Exception:
            pass
        try:
            inputs.pop("token_type_ids", None)
        except Exception:
            pass

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=settings.glm_ocr_max_tokens,
            )

        prompt_len = inputs["input_ids"].shape[1]
        output_text = processor.decode(
            generated_ids[0][prompt_len:],
            skip_special_tokens=False,
        )
        return str(output_text)
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def ocr_image_base64(image_base64: str, *, settings: Settings) -> str:
    """
    Extract Markdown/text from a base64-encoded image using GLM-OCR.

    Modes:
    - transformers (default): run in-process via Hugging Face Transformers.
    - server: call an OpenAI-compatible GLM-OCR server at
      `{RAGPREP_GLM_OCR_BASE_URL}/v1/chat/completions`.
    """

    mode = (settings.glm_ocr_mode or "").strip().lower()
    if not mode:
        mode = _GlmOcrMode.transformers

    if mode == _GlmOcrMode.server:
        return _ocr_image_base64_via_server(image_base64, settings=settings)
    if mode == _GlmOcrMode.transformers:
        return _ocr_image_base64_via_transformers(image_base64, settings=settings)

    raise RuntimeError(f"Unsupported GLM-OCR mode: {settings.glm_ocr_mode!r}")
