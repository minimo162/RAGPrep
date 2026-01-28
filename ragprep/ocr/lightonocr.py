from __future__ import annotations

import base64
import io
import os

from PIL import Image

from ragprep import config as app_config

from .llama_server_client import LlamaServerClientSettings
from .llama_server_client import ocr_image_base64 as ocr_image_base64_llama_server

ENV_MAX_NEW_TOKENS = "LIGHTONOCR_MAX_NEW_TOKENS"
ENV_DRY_RUN = "LIGHTONOCR_DRY_RUN"
ENV_TEMPERATURE = "LIGHTONOCR_TEMPERATURE"
ENV_LIGHTONOCR_LLAMA_SERVER_URL = "LIGHTONOCR_LLAMA_SERVER_URL"
ENV_LIGHTONOCR_MODEL = "LIGHTONOCR_MODEL"

DRY_RUN_OUTPUT = "LIGHTONOCR_DRY_RUN=1 (no inference)"
_DEFAULT_MAX_NEW_TOKENS = 1000
_DEFAULT_TEMPERATURE = 0.2


def _parse_int_env(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    raw = value.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an int, got: {raw!r}") from exc


def _parse_float_env(name: str, *, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    raw = value.strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got: {raw!r}") from exc


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _image_to_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _build_llama_server_settings() -> LlamaServerClientSettings:
    app_settings = app_config.get_settings()
    base_url = app_settings.lightonocr_llama_server_url
    model = app_settings.lightonocr_model
    if not base_url:
        raise ValueError(f"{ENV_LIGHTONOCR_LLAMA_SERVER_URL} is required")
    if not model:
        raise ValueError(f"{ENV_LIGHTONOCR_MODEL} is required")

    max_new_tokens = _parse_int_env(ENV_MAX_NEW_TOKENS, default=_DEFAULT_MAX_NEW_TOKENS)
    temperature = _parse_float_env(ENV_TEMPERATURE, default=_DEFAULT_TEMPERATURE)

    return LlamaServerClientSettings(
        base_url=base_url,
        model=model,
        timeout_seconds=app_settings.lightonocr_request_timeout_seconds,
        max_tokens=max_new_tokens,
        temperature=temperature,
    )


def ocr_image(image: Image.Image) -> str:
    """
    Run LightOnOCR on a single image and return extracted Markdown/text.

    Environment variables:
    - LIGHTONOCR_LLAMA_SERVER_URL: llama-server base URL
    - LIGHTONOCR_MODEL: llama-server model name
    - LIGHTONOCR_REQUEST_TIMEOUT_SECONDS: request timeout in seconds
    - LIGHTONOCR_MAX_NEW_TOKENS: max tokens to generate (default: 1000)
    - LIGHTONOCR_TEMPERATURE: sampling temperature (default: 0.2)
    - LIGHTONOCR_DRY_RUN: if truthy, return a fixed string and do no inference
    """

    if _env_truthy(ENV_DRY_RUN):
        return DRY_RUN_OUTPUT

    server_settings = _build_llama_server_settings()
    image_base64 = _image_to_png_base64(image)
    return ocr_image_base64_llama_server(
        image_base64=image_base64,
        prompt="Extract all text from this image and return it as Markdown.",
        settings=server_settings,
    )


def ocr_image_base64(image_base64: str) -> str:
    """
    Run LightOnOCR on a base64-encoded PNG image and return extracted Markdown/text.

    Environment variables:
    - LIGHTONOCR_LLAMA_SERVER_URL: llama-server base URL
    - LIGHTONOCR_MODEL: llama-server model name
    - LIGHTONOCR_REQUEST_TIMEOUT_SECONDS: request timeout in seconds
    - LIGHTONOCR_MAX_NEW_TOKENS: max tokens to generate (default: 1000)
    - LIGHTONOCR_TEMPERATURE: sampling temperature (default: 0.2)
    - LIGHTONOCR_DRY_RUN: if truthy, return a fixed string and do no inference
    """

    if _env_truthy(ENV_DRY_RUN):
        return DRY_RUN_OUTPUT

    server_settings = _build_llama_server_settings()
    return ocr_image_base64_llama_server(
        image_base64=image_base64,
        prompt="Extract all text from this image and return it as Markdown.",
        settings=server_settings,
    )
