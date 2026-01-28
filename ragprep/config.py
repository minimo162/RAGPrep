from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Final

ENV_MAX_UPLOAD_BYTES: Final[str] = "RAGPREP_MAX_UPLOAD_BYTES"
ENV_MAX_PAGES: Final[str] = "RAGPREP_MAX_PAGES"
ENV_RENDER_DPI: Final[str] = "RAGPREP_RENDER_DPI"
ENV_RENDER_MAX_EDGE: Final[str] = "RAGPREP_RENDER_MAX_EDGE"
ENV_MAX_CONCURRENCY: Final[str] = "RAGPREP_MAX_CONCURRENCY"
ENV_PDF_BACKEND: Final[str] = "RAGPREP_PDF_BACKEND"
ENV_LIGHTONOCR_GGUF_MODEL_PATH: Final[str] = "LIGHTONOCR_GGUF_MODEL_PATH"
ENV_LIGHTONOCR_GGUF_MMPROJ_PATH: Final[str] = "LIGHTONOCR_GGUF_MMPROJ_PATH"
ENV_LIGHTONOCR_LLAVA_CLI_PATH: Final[str] = "LIGHTONOCR_LLAVA_CLI_PATH"
ENV_LIGHTONOCR_BACKEND: Final[str] = "LIGHTONOCR_BACKEND"
ENV_LIGHTONOCR_LLAMA_SERVER_URL: Final[str] = "LIGHTONOCR_LLAMA_SERVER_URL"
ENV_LIGHTONOCR_MODEL: Final[str] = "LIGHTONOCR_MODEL"
ENV_LIGHTONOCR_REQUEST_TIMEOUT_SECONDS: Final[str] = "LIGHTONOCR_REQUEST_TIMEOUT_SECONDS"

_logger = logging.getLogger(__name__)

DEFAULT_MAX_UPLOAD_BYTES: Final[int] = 10 * 1024 * 1024
DEFAULT_MAX_PAGES: Final[int] = 50
DEFAULT_RENDER_DPI: Final[int] = 400
DEFAULT_RENDER_MAX_EDGE: Final[int] = 1540
DEFAULT_MAX_CONCURRENCY: Final[int] = 1
DEFAULT_PDF_BACKEND: Final[str] = "lightonocr"
SUPPORTED_PDF_BACKENDS: Final[tuple[str, ...]] = ("lightonocr",)
DEFAULT_LIGHTONOCR_BACKEND: Final[str] = "cli"
SUPPORTED_LIGHTONOCR_BACKENDS: Final[tuple[str, ...]] = ("cli", "llama-server")
DEFAULT_LIGHTONOCR_LLAMA_SERVER_URL: Final[str] = "http://127.0.0.1:8080"
DEFAULT_LIGHTONOCR_REQUEST_TIMEOUT_SECONDS: Final[int] = 60


@dataclass(frozen=True)
class Settings:
    max_upload_bytes: int
    max_pages: int
    render_dpi: int
    render_max_edge: int
    max_concurrency: int
    pdf_backend: str
    lightonocr_model_path: str | None
    lightonocr_mmproj_path: str | None
    lightonocr_llava_cli_path: str | None
    lightonocr_backend: str
    lightonocr_llama_server_url: str
    lightonocr_model: str | None
    lightonocr_request_timeout_seconds: int


def _get_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an int, got: {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got: {value}")
    return value


def _get_optional_str(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _get_pdf_backend() -> str:
    raw = os.getenv(ENV_PDF_BACKEND)
    if raw is None or not raw.strip():
        return DEFAULT_PDF_BACKEND
    value = raw.strip().lower()
    if value not in SUPPORTED_PDF_BACKENDS:
        raise ValueError(
            f"{ENV_PDF_BACKEND} must be one of {', '.join(SUPPORTED_PDF_BACKENDS)}, got: {raw!r}"
        )
    return value


def _get_lightonocr_backend() -> str:
    raw = os.getenv(ENV_LIGHTONOCR_BACKEND)
    if raw is None or not raw.strip():
        return DEFAULT_LIGHTONOCR_BACKEND
    value = raw.strip().lower().replace("_", "-")
    if value not in SUPPORTED_LIGHTONOCR_BACKENDS:
        raise ValueError(
            f"{ENV_LIGHTONOCR_BACKEND} must be one of "
            f"{', '.join(SUPPORTED_LIGHTONOCR_BACKENDS)}, got: {raw!r}"
        )
    return value


def _warn_if_deprecated_llava_cli_path(value: str | None) -> None:
    if not value:
        return
    _logger.warning(
        "%s is deprecated and will be removed. "
        "Use %s=llama-server with %s/%s instead.",
        ENV_LIGHTONOCR_LLAVA_CLI_PATH,
        ENV_LIGHTONOCR_BACKEND,
        ENV_LIGHTONOCR_LLAMA_SERVER_URL,
        ENV_LIGHTONOCR_MODEL,
    )


def get_settings() -> Settings:
    llava_cli_path = _get_optional_str(ENV_LIGHTONOCR_LLAVA_CLI_PATH)
    _warn_if_deprecated_llava_cli_path(llava_cli_path)
    return Settings(
        max_upload_bytes=_get_positive_int(ENV_MAX_UPLOAD_BYTES, DEFAULT_MAX_UPLOAD_BYTES),
        max_pages=_get_positive_int(ENV_MAX_PAGES, DEFAULT_MAX_PAGES),
        render_dpi=_get_positive_int(ENV_RENDER_DPI, DEFAULT_RENDER_DPI),
        render_max_edge=_get_positive_int(ENV_RENDER_MAX_EDGE, DEFAULT_RENDER_MAX_EDGE),
        max_concurrency=_get_positive_int(ENV_MAX_CONCURRENCY, DEFAULT_MAX_CONCURRENCY),
        pdf_backend=_get_pdf_backend(),
        lightonocr_model_path=_get_optional_str(ENV_LIGHTONOCR_GGUF_MODEL_PATH),
        lightonocr_mmproj_path=_get_optional_str(ENV_LIGHTONOCR_GGUF_MMPROJ_PATH),
        lightonocr_llava_cli_path=llava_cli_path,
        lightonocr_backend=_get_lightonocr_backend(),
        lightonocr_llama_server_url=_get_optional_str(ENV_LIGHTONOCR_LLAMA_SERVER_URL)
        or DEFAULT_LIGHTONOCR_LLAMA_SERVER_URL,
        lightonocr_model=_get_optional_str(ENV_LIGHTONOCR_MODEL),
        lightonocr_request_timeout_seconds=_get_positive_int(
            ENV_LIGHTONOCR_REQUEST_TIMEOUT_SECONDS,
            DEFAULT_LIGHTONOCR_REQUEST_TIMEOUT_SECONDS,
        ),
    )
