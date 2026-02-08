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
ENV_OCR_BACKEND: Final[str] = "RAGPREP_OCR_BACKEND"
ENV_LIGHTON_BASE_URL: Final[str] = "RAGPREP_LIGHTON_BASE_URL"
ENV_LIGHTON_MODEL: Final[str] = "RAGPREP_LIGHTON_MODEL"
ENV_LIGHTON_API_KEY: Final[str] = "RAGPREP_LIGHTON_API_KEY"
ENV_LIGHTON_MAX_TOKENS: Final[str] = "RAGPREP_LIGHTON_MAX_TOKENS"
ENV_LIGHTON_TIMEOUT_SECONDS: Final[str] = "RAGPREP_LIGHTON_TIMEOUT_SECONDS"
ENV_GLM_OCR_BASE_URL: Final[str] = "RAGPREP_GLM_OCR_BASE_URL"
ENV_GLM_OCR_MODEL: Final[str] = "RAGPREP_GLM_OCR_MODEL"
ENV_GLM_OCR_MODE: Final[str] = "RAGPREP_GLM_OCR_MODE"
ENV_GLM_OCR_API_KEY: Final[str] = "RAGPREP_GLM_OCR_API_KEY"
ENV_GLM_OCR_MAX_TOKENS: Final[str] = "RAGPREP_GLM_OCR_MAX_TOKENS"
ENV_GLM_OCR_TIMEOUT_SECONDS: Final[str] = "RAGPREP_GLM_OCR_TIMEOUT_SECONDS"
ENV_LAYOUT_MODE: Final[str] = "RAGPREP_LAYOUT_MODE"
ENV_LAYOUT_BASE_URL: Final[str] = "RAGPREP_LAYOUT_BASE_URL"
ENV_LAYOUT_MODEL: Final[str] = "RAGPREP_LAYOUT_MODEL"
ENV_LAYOUT_API_KEY: Final[str] = "RAGPREP_LAYOUT_API_KEY"
ENV_LAYOUT_MAX_TOKENS: Final[str] = "RAGPREP_LAYOUT_MAX_TOKENS"
ENV_LAYOUT_TIMEOUT_SECONDS: Final[str] = "RAGPREP_LAYOUT_TIMEOUT_SECONDS"
ENV_LAYOUT_RENDER_DPI: Final[str] = "RAGPREP_LAYOUT_RENDER_DPI"
ENV_LAYOUT_RENDER_MAX_EDGE: Final[str] = "RAGPREP_LAYOUT_RENDER_MAX_EDGE"
ENV_LAYOUT_RENDER_AUTO: Final[str] = "RAGPREP_LAYOUT_RENDER_AUTO"
ENV_LAYOUT_RENDER_AUTO_SMALL_DPI: Final[str] = "RAGPREP_LAYOUT_RENDER_AUTO_SMALL_DPI"
ENV_LAYOUT_RENDER_AUTO_SMALL_MAX_EDGE: Final[str] = "RAGPREP_LAYOUT_RENDER_AUTO_SMALL_MAX_EDGE"
ENV_LAYOUT_CONCURRENCY: Final[str] = "RAGPREP_LAYOUT_CONCURRENCY"
ENV_LAYOUT_RETRY_COUNT: Final[str] = "RAGPREP_LAYOUT_RETRY_COUNT"
ENV_LAYOUT_RETRY_BACKOFF_SECONDS: Final[str] = "RAGPREP_LAYOUT_RETRY_BACKOFF_SECONDS"

DEFAULT_MAX_UPLOAD_BYTES: Final[int] = 10 * 1024 * 1024
DEFAULT_MAX_PAGES: Final[int] = 50
DEFAULT_RENDER_DPI: Final[int] = 400
DEFAULT_RENDER_MAX_EDGE: Final[int] = 1540
DEFAULT_MAX_CONCURRENCY: Final[int] = 1
DEFAULT_PDF_BACKEND: Final[str] = "lighton-ocr"
SUPPORTED_PDF_BACKENDS: Final[tuple[str, ...]] = ("lighton-ocr", "glm-ocr")
DEFAULT_LIGHTON_BASE_URL: Final[str] = "http://127.0.0.1:8080"
DEFAULT_LIGHTON_MODEL: Final[str] = "noctrex/LightOnOCR-2-1B-GGUF"
DEFAULT_LIGHTON_MAX_TOKENS: Final[int] = 8192
DEFAULT_LIGHTON_TIMEOUT_SECONDS: Final[int] = 60
DEFAULT_GLM_OCR_BASE_URL: Final[str] = "http://127.0.0.1:8080"
DEFAULT_GLM_OCR_MODEL: Final[str] = "zai-org/GLM-OCR"
DEFAULT_GLM_OCR_MODE: Final[str] = "transformers"
SUPPORTED_GLM_OCR_MODES: Final[tuple[str, ...]] = ("transformers", "server")
SUPPORTED_LAYOUT_MODES: Final[tuple[str, ...]] = (
    "lighton",
    "local-fast",
    "transformers",
    "server",
    "local-paddle",
)
DEFAULT_GLM_OCR_MAX_TOKENS: Final[int] = 8192
DEFAULT_GLM_OCR_TIMEOUT_SECONDS: Final[int] = 60
DEFAULT_LAYOUT_MODE: Final[str] = "lighton"
DEFAULT_LAYOUT_BASE_URL: Final[str] = DEFAULT_LIGHTON_BASE_URL
DEFAULT_LAYOUT_MODEL: Final[str] = DEFAULT_LIGHTON_MODEL
DEFAULT_LAYOUT_MAX_TOKENS: Final[int] = DEFAULT_LIGHTON_MAX_TOKENS
DEFAULT_LAYOUT_TIMEOUT_SECONDS: Final[int] = DEFAULT_LIGHTON_TIMEOUT_SECONDS
DEFAULT_LAYOUT_CONCURRENCY: Final[int] = 1
DEFAULT_LAYOUT_RETRY_COUNT: Final[int] = 1
DEFAULT_LAYOUT_RETRY_BACKOFF_SECONDS: Final[float] = 0.0
DEFAULT_LAYOUT_RENDER_DPI: Final[int] = 250
DEFAULT_LAYOUT_RENDER_MAX_EDGE: Final[int] = 1024
DEFAULT_LAYOUT_RENDER_AUTO: Final[bool] = False
DEFAULT_LAYOUT_RENDER_AUTO_SMALL_DPI: Final[int] = 250
DEFAULT_LAYOUT_RENDER_AUTO_SMALL_MAX_EDGE: Final[int] = 1024


@dataclass(frozen=True)
class Settings:
    max_upload_bytes: int
    max_pages: int
    render_dpi: int
    render_max_edge: int
    max_concurrency: int
    ocr_backend: str
    pdf_backend: str
    lighton_base_url: str
    lighton_model: str
    lighton_api_key: str | None
    lighton_max_tokens: int
    lighton_timeout_seconds: int
    glm_ocr_base_url: str
    glm_ocr_model: str
    glm_ocr_mode: str
    glm_ocr_api_key: str | None
    glm_ocr_max_tokens: int
    glm_ocr_timeout_seconds: int
    layout_mode: str
    layout_base_url: str
    layout_model: str
    layout_api_key: str | None
    layout_max_tokens: int
    layout_timeout_seconds: int
    layout_render_dpi: int
    layout_render_max_edge: int
    layout_render_auto: bool
    layout_render_auto_small_dpi: int
    layout_render_auto_small_max_edge: int
    layout_concurrency: int
    layout_retry_count: int
    layout_retry_backoff_seconds: float


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


def _get_nonnegative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an int, got: {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got: {value}")
    return value


def _get_nonnegative_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got: {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got: {value}")
    return value


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a bool (0/1/true/false), got: {raw!r}")


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
    raw = os.getenv(ENV_OCR_BACKEND)
    if raw is None or not raw.strip():
        raw = os.getenv(ENV_PDF_BACKEND)
    if raw is None or not raw.strip():
        return DEFAULT_PDF_BACKEND
    value = raw.strip().lower()
    if value not in SUPPORTED_PDF_BACKENDS:
        raise ValueError(
            f"{ENV_PDF_BACKEND} must be one of {', '.join(SUPPORTED_PDF_BACKENDS)}, got: {raw!r}"
        )
    return value


def _get_lighton_base_url() -> str:
    raw = os.getenv(ENV_LIGHTON_BASE_URL)
    if raw is None or not raw.strip():
        return DEFAULT_LIGHTON_BASE_URL
    return raw.strip()


def _get_lighton_model() -> str:
    raw = os.getenv(ENV_LIGHTON_MODEL)
    if raw is None or not raw.strip():
        return DEFAULT_LIGHTON_MODEL
    return raw.strip()


def _get_lighton_api_key() -> str | None:
    return _get_optional_str(ENV_LIGHTON_API_KEY)


def _get_lighton_max_tokens() -> int:
    return _get_positive_int(ENV_LIGHTON_MAX_TOKENS, DEFAULT_LIGHTON_MAX_TOKENS)


def _get_lighton_timeout_seconds() -> int:
    return _get_positive_int(ENV_LIGHTON_TIMEOUT_SECONDS, DEFAULT_LIGHTON_TIMEOUT_SECONDS)


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


def _normalize_layout_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "transformers":
        # Backward-compatible alias.
        return "local-fast"
    return normalized


def _get_layout_mode() -> str:
    raw = os.getenv(ENV_LAYOUT_MODE)
    if raw is None or not raw.strip():
        return DEFAULT_LAYOUT_MODE

    value = _normalize_layout_mode(raw)
    if value not in SUPPORTED_LAYOUT_MODES:
        raise ValueError(
            f"{ENV_LAYOUT_MODE} must be one of {', '.join(SUPPORTED_LAYOUT_MODES)}, got: {raw!r}"
        )
    return value


def _get_layout_base_url() -> str:
    raw = os.getenv(ENV_LAYOUT_BASE_URL)
    if raw is not None and raw.strip():
        return raw.strip()

    mode = _get_layout_mode()
    if mode == "lighton":
        return _get_lighton_base_url()
    if _get_pdf_backend() == "glm-ocr":
        return _get_trimmed_str(ENV_GLM_OCR_BASE_URL, DEFAULT_GLM_OCR_BASE_URL)
    return _get_lighton_base_url()


def _get_layout_model() -> str:
    raw = os.getenv(ENV_LAYOUT_MODEL)
    if raw is not None and raw.strip():
        return raw.strip()

    mode = _get_layout_mode()
    if mode == "lighton":
        return _get_lighton_model()
    if _get_pdf_backend() == "glm-ocr":
        return _get_trimmed_str(ENV_GLM_OCR_MODEL, DEFAULT_GLM_OCR_MODEL)
    return _get_lighton_model()


def _get_layout_api_key() -> str | None:
    raw = os.getenv(ENV_LAYOUT_API_KEY)
    if raw is None:
        if _get_layout_mode() == "lighton":
            return _get_lighton_api_key()
        return _get_optional_str(ENV_GLM_OCR_API_KEY)
    value = raw.strip()
    return value or None


def _get_layout_max_tokens() -> int:
    raw = os.getenv(ENV_LAYOUT_MAX_TOKENS)
    if raw is None or not raw.strip():
        if _get_layout_mode() == "lighton":
            return _get_lighton_max_tokens()
        return _get_positive_int(ENV_GLM_OCR_MAX_TOKENS, DEFAULT_GLM_OCR_MAX_TOKENS)
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{ENV_LAYOUT_MAX_TOKENS} must be an int, got: {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{ENV_LAYOUT_MAX_TOKENS} must be > 0, got: {value}")
    return value


def _get_layout_timeout_seconds() -> int:
    raw = os.getenv(ENV_LAYOUT_TIMEOUT_SECONDS)
    if raw is None or not raw.strip():
        if _get_layout_mode() == "lighton":
            return _get_lighton_timeout_seconds()
        return _get_positive_int(ENV_GLM_OCR_TIMEOUT_SECONDS, DEFAULT_GLM_OCR_TIMEOUT_SECONDS)
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{ENV_LAYOUT_TIMEOUT_SECONDS} must be an int, got: {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{ENV_LAYOUT_TIMEOUT_SECONDS} must be > 0, got: {value}")
    return value


def _get_layout_render_dpi() -> int:
    raw = os.getenv(ENV_LAYOUT_RENDER_DPI)
    if raw is None or not raw.strip():
        return DEFAULT_LAYOUT_RENDER_DPI
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{ENV_LAYOUT_RENDER_DPI} must be an int, got: {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{ENV_LAYOUT_RENDER_DPI} must be > 0, got: {value}")
    return value


def _get_layout_render_max_edge() -> int:
    raw = os.getenv(ENV_LAYOUT_RENDER_MAX_EDGE)
    if raw is None or not raw.strip():
        return DEFAULT_LAYOUT_RENDER_MAX_EDGE
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{ENV_LAYOUT_RENDER_MAX_EDGE} must be an int, got: {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{ENV_LAYOUT_RENDER_MAX_EDGE} must be > 0, got: {value}")
    return value


def _get_layout_render_auto() -> bool:
    return _get_bool(ENV_LAYOUT_RENDER_AUTO, DEFAULT_LAYOUT_RENDER_AUTO)


def _get_layout_render_auto_small_dpi() -> int:
    return _get_positive_int(ENV_LAYOUT_RENDER_AUTO_SMALL_DPI, DEFAULT_LAYOUT_RENDER_AUTO_SMALL_DPI)


def _get_layout_render_auto_small_max_edge() -> int:
    return _get_positive_int(
        ENV_LAYOUT_RENDER_AUTO_SMALL_MAX_EDGE, DEFAULT_LAYOUT_RENDER_AUTO_SMALL_MAX_EDGE
    )


def _get_layout_concurrency() -> int:
    return _get_positive_int(ENV_LAYOUT_CONCURRENCY, DEFAULT_LAYOUT_CONCURRENCY)


def _get_layout_retry_count() -> int:
    return _get_nonnegative_int(ENV_LAYOUT_RETRY_COUNT, DEFAULT_LAYOUT_RETRY_COUNT)


def _get_layout_retry_backoff_seconds() -> float:
    return _get_nonnegative_float(
        ENV_LAYOUT_RETRY_BACKOFF_SECONDS, DEFAULT_LAYOUT_RETRY_BACKOFF_SECONDS
    )


def get_settings() -> Settings:
    ocr_backend = _get_pdf_backend()
    return Settings(
        max_upload_bytes=_get_positive_int(ENV_MAX_UPLOAD_BYTES, DEFAULT_MAX_UPLOAD_BYTES),
        max_pages=_get_positive_int(ENV_MAX_PAGES, DEFAULT_MAX_PAGES),
        render_dpi=_get_positive_int(ENV_RENDER_DPI, DEFAULT_RENDER_DPI),
        render_max_edge=_get_positive_int(ENV_RENDER_MAX_EDGE, DEFAULT_RENDER_MAX_EDGE),
        max_concurrency=_get_positive_int(ENV_MAX_CONCURRENCY, DEFAULT_MAX_CONCURRENCY),
        ocr_backend=ocr_backend,
        pdf_backend=ocr_backend,
        lighton_base_url=_get_lighton_base_url(),
        lighton_model=_get_lighton_model(),
        lighton_api_key=_get_lighton_api_key(),
        lighton_max_tokens=_get_lighton_max_tokens(),
        lighton_timeout_seconds=_get_lighton_timeout_seconds(),
        glm_ocr_base_url=_get_trimmed_str(ENV_GLM_OCR_BASE_URL, DEFAULT_GLM_OCR_BASE_URL),
        glm_ocr_model=_get_trimmed_str(ENV_GLM_OCR_MODEL, DEFAULT_GLM_OCR_MODEL),
        glm_ocr_mode=_get_glm_ocr_mode(),
        glm_ocr_api_key=_get_optional_str(ENV_GLM_OCR_API_KEY),
        glm_ocr_max_tokens=_get_positive_int(ENV_GLM_OCR_MAX_TOKENS, DEFAULT_GLM_OCR_MAX_TOKENS),
        glm_ocr_timeout_seconds=_get_positive_int(
            ENV_GLM_OCR_TIMEOUT_SECONDS, DEFAULT_GLM_OCR_TIMEOUT_SECONDS
        ),
        layout_mode=_get_layout_mode(),
        layout_base_url=_get_layout_base_url(),
        layout_model=_get_layout_model(),
        layout_api_key=_get_layout_api_key(),
        layout_max_tokens=_get_layout_max_tokens(),
        layout_timeout_seconds=_get_layout_timeout_seconds(),
        layout_render_dpi=_get_layout_render_dpi(),
        layout_render_max_edge=_get_layout_render_max_edge(),
        layout_render_auto=_get_layout_render_auto(),
        layout_render_auto_small_dpi=_get_layout_render_auto_small_dpi(),
        layout_render_auto_small_max_edge=_get_layout_render_auto_small_max_edge(),
        layout_concurrency=_get_layout_concurrency(),
        layout_retry_count=_get_layout_retry_count(),
        layout_retry_backoff_seconds=_get_layout_retry_backoff_seconds(),
    )



