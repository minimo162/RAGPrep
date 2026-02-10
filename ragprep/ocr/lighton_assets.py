from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import httpx

from ragprep.config import Settings

_GLOBAL_LOCK = Lock()
_PATH_LOCKS: dict[str, Lock] = {}

_HF_TOKEN_ENV_KEYS: tuple[str, ...] = ("RAGPREP_HF_TOKEN", "HF_TOKEN")


@dataclass(frozen=True)
class LightOnAssetPaths:
    model_path: Path
    mmproj_path: Path


def ensure_lighton_assets(
    settings: Settings,
    *,
    mmproj_file: str | None = None,
) -> LightOnAssetPaths:
    root = Path(settings.lighton_model_dir).expanduser()
    model_path = root / settings.lighton_model_file
    mmproj_name = str(mmproj_file or settings.lighton_mmproj_file).strip()
    if not mmproj_name:
        raise RuntimeError("LightOn mmproj filename must be non-empty.")
    mmproj_path = root / mmproj_name
    root.mkdir(parents=True, exist_ok=True)

    missing = [p for p in (model_path, mmproj_path) if not p.exists()]
    if not missing:
        return LightOnAssetPaths(model_path=model_path, mmproj_path=mmproj_path)

    if not settings.lighton_auto_download:
        missing_text = ", ".join(str(p) for p in missing)
        raise RuntimeError(
            "LightOn model files are missing and auto-download is disabled: "
            f"{missing_text}"
        )

    token = _resolve_hf_token()
    for path in missing:
        lock = _path_lock(path)
        with lock:
            if path.exists():
                continue
            _download_hf_file(
                repo_id=settings.lighton_repo_id,
                filename=path.name,
                target_path=path,
                token=token,
                timeout_seconds=float(settings.lighton_request_timeout_seconds),
            )

    return LightOnAssetPaths(model_path=model_path, mmproj_path=mmproj_path)


def _resolve_hf_token() -> str | None:
    for key in _HF_TOKEN_ENV_KEYS:
        raw = os.getenv(key)
        if raw is not None and raw.strip():
            return raw.strip()
    return None


def _path_lock(path: Path) -> Lock:
    key = str(path.resolve()).lower()
    with _GLOBAL_LOCK:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = Lock()
            _PATH_LOCKS[key] = lock
        return lock


def _download_hf_file(
    *,
    repo_id: str,
    filename: str,
    target_path: Path,
    token: str | None,
    timeout_seconds: float,
) -> None:
    if not repo_id.strip():
        raise RuntimeError("RAGPREP_LIGHTON_REPO_ID must be non-empty.")
    if not filename.strip():
        raise RuntimeError("LightOn filename must be non-empty.")

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}?download=true"
    headers: dict[str, str] = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"

    tmp_path = target_path.with_name(target_path.name + ".part")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with httpx.stream(
            "GET",
            url,
            headers=headers,
            follow_redirects=True,
            timeout=timeout_seconds,
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"Failed to download Hugging Face asset {filename!r}: "
                    f"HTTP {response.status_code}"
                ) from exc

            with tmp_path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    if chunk:
                        handle.write(chunk)

        if not tmp_path.exists() or tmp_path.stat().st_size <= 0:
            raise RuntimeError(f"Downloaded file is empty: {filename!r}")
        os.replace(tmp_path, target_path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise
