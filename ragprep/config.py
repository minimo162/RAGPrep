from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

ENV_MAX_UPLOAD_BYTES: Final[str] = "RAGPREP_MAX_UPLOAD_BYTES"
ENV_MAX_PAGES: Final[str] = "RAGPREP_MAX_PAGES"
ENV_RENDER_DPI: Final[str] = "RAGPREP_RENDER_DPI"
ENV_RENDER_MAX_EDGE: Final[str] = "RAGPREP_RENDER_MAX_EDGE"
ENV_MAX_CONCURRENCY: Final[str] = "RAGPREP_MAX_CONCURRENCY"
ENV_PDF_BACKEND: Final[str] = "RAGPREP_PDF_BACKEND"
ENV_GLM_OCR_BASE_URL: Final[str] = "RAGPREP_GLM_OCR_BASE_URL"
ENV_GLM_OCR_MODEL: Final[str] = "RAGPREP_GLM_OCR_MODEL"
ENV_GLM_OCR_MODE: Final[str] = "RAGPREP_GLM_OCR_MODE"
ENV_GLM_OCR_API_KEY: Final[str] = "RAGPREP_GLM_OCR_API_KEY"
ENV_GLM_OCR_MAX_TOKENS: Final[str] = "RAGPREP_GLM_OCR_MAX_TOKENS"
ENV_GLM_OCR_TIMEOUT_SECONDS: Final[str] = "RAGPREP_GLM_OCR_TIMEOUT_SECONDS"

DEFAULT_MAX_UPLOAD_BYTES: Final[int] = 10 * 1024 * 1024
DEFAULT_MAX_PAGES: Final[int] = 50
DEFAULT_RENDER_DPI: Final[int] = 400
DEFAULT_RENDER_MAX_EDGE: Final[int] = 1540
DEFAULT_MAX_CONCURRENCY: Final[int] = 1
DEFAULT_PDF_BACKEND: Final[str] = "glm-ocr"
SUPPORTED_PDF_BACKENDS: Final[tuple[str, ...]] = ("glm-ocr",)
DEFAULT_GLM_OCR_BASE_URL: Final[str] = "http://127.0.0.1:8080"
DEFAULT_GLM_OCR_MODEL: Final[str] = "zai-org/GLM-OCR"
DEFAULT_GLM_OCR_MODE: Final[str] = "transformers"
SUPPORTED_GLM_OCR_MODES: Final[tuple[str, ...]] = ("transformers", "server")
DEFAULT_GLM_OCR_MAX_TOKENS: Final[int] = 8192
DEFAULT_GLM_OCR_TIMEOUT_SECONDS: Final[int] = 60


@dataclass(frozen=True)
class Settings:
    max_upload_bytes: int
    max_pages: int
    render_dpi: int
    render_max_edge: int
    max_concurrency: int
    pdf_backend: str
    glm_ocr_base_url: str
    glm_ocr_model: str
    glm_ocr_mode: str
    glm_ocr_api_key: str | None
    glm_ocr_max_tokens: int
    glm_ocr_timeout_seconds: int


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


def _get_trimmed_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


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


def _get_glm_ocr_mode() -> str:
    raw = os.getenv(ENV_GLM_OCR_MODE)
    if raw is None or not raw.strip():
        return DEFAULT_GLM_OCR_MODE
    value = raw.strip().lower()
    if value not in SUPPORTED_GLM_OCR_MODES:
        raise ValueError(
            f"{ENV_GLM_OCR_MODE} must be one of {', '.join(SUPPORTED_GLM_OCR_MODES)}, got: {raw!r}"
        )
    return value


def get_settings() -> Settings:
    return Settings(
        max_upload_bytes=_get_positive_int(ENV_MAX_UPLOAD_BYTES, DEFAULT_MAX_UPLOAD_BYTES),
        max_pages=_get_positive_int(ENV_MAX_PAGES, DEFAULT_MAX_PAGES),
        render_dpi=_get_positive_int(ENV_RENDER_DPI, DEFAULT_RENDER_DPI),
        render_max_edge=_get_positive_int(ENV_RENDER_MAX_EDGE, DEFAULT_RENDER_MAX_EDGE),
        max_concurrency=_get_positive_int(ENV_MAX_CONCURRENCY, DEFAULT_MAX_CONCURRENCY),
        pdf_backend=_get_pdf_backend(),
        glm_ocr_base_url=_get_trimmed_str(ENV_GLM_OCR_BASE_URL, DEFAULT_GLM_OCR_BASE_URL),
        glm_ocr_model=_get_trimmed_str(ENV_GLM_OCR_MODEL, DEFAULT_GLM_OCR_MODEL),
        glm_ocr_mode=_get_glm_ocr_mode(),
        glm_ocr_api_key=_get_optional_str(ENV_GLM_OCR_API_KEY),
        glm_ocr_max_tokens=_get_positive_int(ENV_GLM_OCR_MAX_TOKENS, DEFAULT_GLM_OCR_MAX_TOKENS),
        glm_ocr_timeout_seconds=_get_positive_int(
            ENV_GLM_OCR_TIMEOUT_SECONDS, DEFAULT_GLM_OCR_TIMEOUT_SECONDS
        ),
    )
