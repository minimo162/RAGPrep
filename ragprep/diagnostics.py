from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DIR_NAME = "ragprep-diagnostics"


def diagnostics_dir() -> Path:
    raw = os.getenv("RAGPREP_DIAGNOSTICS_DIR")
    if raw and raw.strip():
        return Path(raw.strip()).expanduser()
    return Path(tempfile.gettempdir()) / _DEFAULT_DIR_NAME


def enable_faulthandler() -> bool:
    try:
        import faulthandler

        faulthandler.enable(all_threads=True)
        return True
    except Exception:  # noqa: BLE001
        logger.debug("Failed to enable faulthandler", exc_info=True)
        return False


def sha256_text(value: str, *, chunk_size: int = 1_048_576) -> str:
    if not value:
        return sha256(b"").hexdigest()

    hasher = sha256()
    for offset in range(0, len(value), chunk_size):
        chunk = value[offset : offset + chunk_size]
        hasher.update(chunk.encode("utf-8", errors="ignore"))
    return hasher.hexdigest()


def summarize_base64(value: str) -> dict[str, Any]:
    raw = value or ""
    length = len(raw)
    return {
        "base64_len": length,
        "base64_sha256": sha256_text(raw),
        "estimated_bytes": (length * 3) // 4,
    }


def _safe_write_text(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to write diagnostics artifact: %s", path, exc_info=True)


def write_json_artifact(filename: str, payload: Mapping[str, Any]) -> Path:
    path = diagnostics_dir() / filename
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2)
    _safe_write_text(path, text + "\n")
    return path


def record_last_activity(payload: Mapping[str, Any]) -> Path:
    data: dict[str, Any] = {"ts": time.time(), **dict(payload)}
    return write_json_artifact("last_activity.json", data)


def record_last_llama_request(payload: Mapping[str, Any]) -> Path:
    data: dict[str, Any] = {"ts": time.time(), **dict(payload)}
    return write_json_artifact("last_llama_request.json", data)


def record_last_error(payload: Mapping[str, Any]) -> Path:
    data: dict[str, Any] = {"ts": time.time(), **dict(payload)}
    return write_json_artifact("last_error.json", data)
