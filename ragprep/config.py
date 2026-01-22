from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

ENV_MAX_UPLOAD_BYTES: Final[str] = "RAGPREP_MAX_UPLOAD_BYTES"
ENV_MAX_PAGES: Final[str] = "RAGPREP_MAX_PAGES"
ENV_RENDER_DPI: Final[str] = "RAGPREP_RENDER_DPI"
ENV_MAX_CONCURRENCY: Final[str] = "RAGPREP_MAX_CONCURRENCY"

DEFAULT_MAX_UPLOAD_BYTES: Final[int] = 10 * 1024 * 1024
DEFAULT_MAX_PAGES: Final[int] = 50
DEFAULT_RENDER_DPI: Final[int] = 200
DEFAULT_MAX_CONCURRENCY: Final[int] = 1


@dataclass(frozen=True)
class Settings:
    max_upload_bytes: int
    max_pages: int
    render_dpi: int
    max_concurrency: int


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


def get_settings() -> Settings:
    return Settings(
        max_upload_bytes=_get_positive_int(ENV_MAX_UPLOAD_BYTES, DEFAULT_MAX_UPLOAD_BYTES),
        max_pages=_get_positive_int(ENV_MAX_PAGES, DEFAULT_MAX_PAGES),
        render_dpi=_get_positive_int(ENV_RENDER_DPI, DEFAULT_RENDER_DPI),
        max_concurrency=_get_positive_int(ENV_MAX_CONCURRENCY, DEFAULT_MAX_CONCURRENCY),
    )
