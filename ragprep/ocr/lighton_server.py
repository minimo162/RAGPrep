from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import httpx

from ragprep.config import Settings
from ragprep.ocr.lighton_assets import LightOnAssetPaths, ensure_lighton_assets

_SERVER_LOCK = Lock()
_ACTIVE_PROCESS: subprocess.Popen[bytes] | None = None
_ACTIVE_BASE_URL: str | None = None

_AUTO_DOWNLOAD_ENV = "RAGPREP_LLAMA_SERVER_AUTO_DOWNLOAD"
_INSTALL_DIR_ENV = "RAGPREP_LLAMA_SERVER_INSTALL_DIR"
_GITHUB_RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
_MMPROJ_FALLBACK_FILES: tuple[str, ...] = ("mmproj-F32.gguf",)


@dataclass(frozen=True)
class _StartAttempt:
    name: str
    ngl: int
    parallel: int
    flash_attn: bool


def ensure_server_base_url(settings: Settings) -> str:
    base_url = _base_url(settings)
    with _SERVER_LOCK:
        global _ACTIVE_BASE_URL, _ACTIVE_PROCESS

        if _ACTIVE_PROCESS is not None and _ACTIVE_PROCESS.poll() is None:
            if _is_server_healthy(base_url):
                _ACTIVE_BASE_URL = base_url
                return base_url
            _terminate_process(_ACTIVE_PROCESS)
            _ACTIVE_PROCESS = None
            _ACTIVE_BASE_URL = None
        elif _ACTIVE_PROCESS is not None:
            _ACTIVE_PROCESS = None
            _ACTIVE_BASE_URL = None

        if _is_server_healthy(base_url):
            _ACTIVE_BASE_URL = base_url
            return base_url

        assets_by_mmproj, asset_errors = _resolve_assets_by_mmproj(settings)
        if not assets_by_mmproj:
            joined = "; ".join(asset_errors) if asset_errors else "unknown error"
            raise RuntimeError(f"Failed to resolve LightOn assets. {joined}")
        executable = _resolve_llama_server_executable(settings)
        attempts = _build_attempts(settings)

        errors: list[str] = list(asset_errors)
        for mmproj_name, assets in assets_by_mmproj:
            for attempt in attempts:
                process = _start_llama_server_process(
                    executable=executable,
                    assets=assets,
                    settings=settings,
                    attempt=attempt,
                )
                ok, error = _wait_for_server_ready(
                    process=process,
                    base_url=base_url,
                    timeout_seconds=float(settings.lighton_start_timeout_seconds),
                )
                if ok:
                    _ACTIVE_PROCESS = process
                    _ACTIVE_BASE_URL = base_url
                    return base_url

                _terminate_process(process)
                if error is not None:
                    errors.append(f"mmproj={mmproj_name}/{attempt.name}: {error}")
                else:
                    errors.append(f"mmproj={mmproj_name}/{attempt.name}: unknown error")

        joined = "; ".join(errors) if errors else "unknown error"
        raise RuntimeError(f"Failed to start llama-server after retries. {joined}")


def prewarm_lighton_server(*, settings: Settings) -> None:
    _ = ensure_server_base_url(settings)


def _build_mmproj_candidates(settings: Settings) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    candidates = [str(settings.lighton_mmproj_file or "").strip(), *_MMPROJ_FALLBACK_FILES]
    for raw in candidates:
        name = str(raw or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(name)
    return unique


def _resolve_assets_by_mmproj(
    settings: Settings,
) -> tuple[list[tuple[str, LightOnAssetPaths]], list[str]]:
    assets_by_mmproj: list[tuple[str, LightOnAssetPaths]] = []
    errors: list[str] = []
    for mmproj_name in _build_mmproj_candidates(settings):
        try:
            assets = ensure_lighton_assets(settings, mmproj_file=mmproj_name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"mmproj={mmproj_name}: {exc}")
            continue
        assets_by_mmproj.append((mmproj_name, assets))
    return assets_by_mmproj, errors


def _build_attempts(settings: Settings) -> list[_StartAttempt]:
    attempts = [
        _StartAttempt(
            name="gpu-aggressive",
            ngl=int(settings.lighton_n_gpu_layers),
            parallel=int(settings.lighton_parallel),
            flash_attn=bool(settings.lighton_flash_attn),
        ),
        _StartAttempt(
            name="gpu-conservative",
            ngl=int(settings.lighton_n_gpu_layers),
            parallel=int(settings.lighton_parallel),
            flash_attn=False,
        ),
        _StartAttempt(
            name="cpu-fallback",
            ngl=0,
            parallel=1,
            flash_attn=False,
        ),
    ]

    unique: list[_StartAttempt] = []
    seen: set[tuple[int, int, bool]] = set()
    for attempt in attempts:
        key = (attempt.ngl, attempt.parallel, attempt.flash_attn)
        if key in seen:
            continue
        seen.add(key)
        unique.append(attempt)
    return unique


def _resolve_llama_server_executable(settings: Settings) -> str:
    configured = settings.llama_server_path
    if configured is not None:
        raw = configured.strip()
        if not raw:
            raise RuntimeError("RAGPREP_LLAMA_SERVER_PATH is empty.")
        path = Path(raw).expanduser()
        if path.exists():
            return str(path)
        raise RuntimeError(f"llama-server not found at RAGPREP_LLAMA_SERVER_PATH: {path}")

    discovered = _auto_discover_llama_server_executable(settings)
    if discovered is not None:
        os.environ.setdefault("RAGPREP_LLAMA_SERVER_PATH", discovered)
        return discovered

    installed = _auto_install_llama_server_executable(settings)
    if installed is not None:
        os.environ.setdefault("RAGPREP_LLAMA_SERVER_PATH", installed)
        return installed

    raise RuntimeError(
        "llama-server not found. Install llama.cpp server binary and ensure it is in PATH "
        "or set RAGPREP_LLAMA_SERVER_PATH. "
        "Auto-download was attempted only when enabled and supported."
    )


def _auto_discover_llama_server_executable(settings: Settings) -> str | None:
    command_names = ("llama-server", "llama-server.exe")
    for name in command_names:
        found = shutil.which(name)
        if found:
            return found

    for candidate in _candidate_llama_server_paths(settings):
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError:
            continue
    return None


def _candidate_llama_server_paths(settings: Settings) -> list[Path]:
    binary_names = ("llama-server.exe", "llama-server")
    repo_root = Path(__file__).resolve().parents[2]
    cwd = Path.cwd()
    home = Path.home()
    downloads = home / "Downloads"
    model_dir_parent = Path(settings.lighton_model_dir).expanduser().parent

    base_dirs = [
        cwd,
        repo_root,
        repo_root / "bin",
        repo_root / "tools",
        home,
        home / "llama.cpp",
        home / "tools",
        downloads,
        Path("C:/tools"),
        Path("C:/llama.cpp"),
        Path("C:/Program Files/llama.cpp"),
        model_dir_parent,
    ]
    relative_dirs = (
        Path("."),
        Path("bin"),
        Path("build/bin"),
        Path("build/bin/Release"),
        Path("llama.cpp"),
        Path("llama.cpp/bin"),
        Path("llama.cpp/build/bin"),
        Path("llama.cpp/build/bin/Release"),
    )

    candidates: list[Path] = []
    seen: set[str] = set()

    def _push(path: Path) -> None:
        key = str(path).lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    for base in base_dirs:
        for relative in relative_dirs:
            folder = (base / relative) if relative != Path(".") else base
            for binary_name in binary_names:
                _push(folder / binary_name)

    if downloads.is_dir():
        try:
            for child in downloads.iterdir():
                if not child.is_dir():
                    continue
                for relative in (
                    Path("."),
                    Path("bin"),
                    Path("build/bin"),
                    Path("build/bin/Release"),
                ):
                    folder = (child / relative) if relative != Path(".") else child
                    for binary_name in binary_names:
                        _push(folder / binary_name)
        except OSError:
            pass

    return candidates


def _auto_install_llama_server_executable(settings: Settings) -> str | None:
    if not _is_auto_download_enabled():
        return None

    install_root = _llama_server_install_root(settings)
    existing = _find_llama_server_in_dir(install_root)
    if existing is not None:
        return str(existing)

    asset = _select_prebuilt_asset(_fetch_latest_release_assets(settings))
    if asset is None:
        return None

    asset_name = str(asset.get("name", "")).strip()
    download_url = str(asset.get("browser_download_url", "")).strip()
    if not asset_name or not download_url:
        return None

    install_root.mkdir(parents=True, exist_ok=True)
    version_dir = install_root / _safe_asset_stem(asset_name)
    archive_path = install_root / asset_name
    _download_release_asset(
        download_url=download_url,
        target_path=archive_path,
        timeout_seconds=float(settings.lighton_request_timeout_seconds),
    )
    try:
        _extract_archive(archive_path=archive_path, target_dir=version_dir)
    finally:
        try:
            if archive_path.exists():
                archive_path.unlink()
        except OSError:
            pass

    installed = _find_llama_server_in_dir(version_dir)
    if installed is None:
        return None
    return str(installed)


def _is_auto_download_enabled() -> bool:
    raw = os.getenv(_AUTO_DOWNLOAD_ENV)
    if raw is None or not raw.strip():
        return True
    value = raw.strip().lower()
    return value in {"1", "true", "yes", "on"}


def _llama_server_install_root(settings: Settings) -> Path:
    raw = os.getenv(_INSTALL_DIR_ENV)
    if raw is not None and raw.strip():
        return Path(raw.strip()).expanduser()
    return Path(settings.lighton_model_dir).expanduser().parent / "llama-server"


def _safe_asset_stem(name: str) -> str:
    stem = name
    for suffix in (".tar.gz", ".tgz", ".zip"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return Path(stem).stem


def _find_llama_server_in_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    is_windows = sys.platform.startswith("win")
    candidates = ("llama-server.exe",) if is_windows else ("llama-server", "llama-server.exe")
    for filename in candidates:
        direct = root / filename
        if direct.is_file():
            return direct
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if lowered == "llama-server.exe":
            return path
        if lowered == "llama-server" and not is_windows:
            return path
    return None


def _fetch_latest_release_assets(settings: Settings) -> list[dict[str, object]]:
    try:
        with httpx.Client(
            timeout=float(settings.lighton_request_timeout_seconds),
            trust_env=False,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "ragprep",
            },
        ) as client:
            response = client.get(_GITHUB_RELEASE_API)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    if not isinstance(payload, dict):
        return []
    assets = payload.get("assets")
    if not isinstance(assets, list):
        return []

    out: list[dict[str, object]] = []
    for item in assets:
        if isinstance(item, dict):
            out.append(item)
    return out


def _select_prebuilt_asset(
    assets: list[dict[str, object]],
    *,
    platform_name: str | None = None,
) -> dict[str, object] | None:
    if not assets:
        return None
    platform = (platform_name or sys.platform).lower()

    typed_assets: list[dict[str, object]] = []
    for asset in assets:
        name = str(asset.get("name", "")).lower()
        url = str(asset.get("browser_download_url", "")).strip()
        if not name or not url:
            continue
        if not (name.endswith(".zip") or name.endswith(".tar.gz") or name.endswith(".tgz")):
            continue
        typed_assets.append(asset)
    if not typed_assets:
        return None

    if platform.startswith("win"):
        return _pick_asset_by_preferences(
            typed_assets,
            include_all=("win", "x64"),
            preferred=(
                "win-avx2-x64",
                "win-avx-x64",
                "win-openblas-x64",
                "win-noavx-x64",
            ),
            excluded=("cuda", "cudart", "vulkan", "sycl", "hip", "rocm", "metal"),
        )
    if platform.startswith("linux"):
        return _pick_asset_by_preferences(
            typed_assets,
            include_all=("linux", "x64"),
            preferred=("linux-avx2-x64", "linux-avx-x64", "linux-x64"),
            excluded=("cuda", "vulkan", "sycl", "hip", "rocm", "metal"),
        )
    if platform == "darwin":
        return _pick_asset_by_preferences(
            typed_assets,
            include_all=("macos", "arm64"),
            preferred=("macos-arm64",),
            excluded=("cuda", "vulkan", "sycl", "hip", "rocm"),
        )
    return None


def _pick_asset_by_preferences(
    assets: list[dict[str, object]],
    *,
    include_all: tuple[str, ...],
    preferred: tuple[str, ...],
    excluded: tuple[str, ...],
) -> dict[str, object] | None:
    filtered: list[dict[str, object]] = []
    for asset in assets:
        name = str(asset.get("name", "")).lower()
        if any(token not in name for token in include_all):
            continue
        if any(token in name for token in excluded):
            continue
        filtered.append(asset)
    if not filtered:
        return None

    for key in preferred:
        for asset in filtered:
            name = str(asset.get("name", "")).lower()
            if key in name:
                return asset
    return filtered[0]


def _download_release_asset(
    *,
    download_url: str,
    target_path: Path,
    timeout_seconds: float,
) -> None:
    tmp_path = target_path.with_name(target_path.name + ".part")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.stream(
            "GET",
            download_url,
            follow_redirects=True,
            timeout=timeout_seconds,
        ) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    if chunk:
                        handle.write(chunk)
        if not tmp_path.exists() or tmp_path.stat().st_size <= 0:
            raise RuntimeError(f"Downloaded llama-server archive is empty: {download_url}")
        os.replace(tmp_path, target_path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def _extract_archive(*, archive_path: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    lower_name = archive_path.name.lower()
    if lower_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, mode="r") as zf:
            zf.extractall(target_dir)
        return
    if lower_name.endswith(".tar.gz") or lower_name.endswith(".tgz"):
        with tarfile.open(archive_path, mode="r:gz") as tf:
            tf.extractall(target_dir)
        return
    raise RuntimeError(f"Unsupported archive format: {archive_path.name}")


def _base_url(settings: Settings) -> str:
    return f"http://{settings.lighton_server_host}:{settings.lighton_server_port}"


def _start_llama_server_process(
    *,
    executable: str,
    assets: LightOnAssetPaths,
    settings: Settings,
    attempt: _StartAttempt,
) -> subprocess.Popen[bytes]:
    command = [
        executable,
        "-m",
        str(assets.model_path),
        "--mmproj",
        str(assets.mmproj_path),
        "--host",
        settings.lighton_server_host,
        "--port",
        str(settings.lighton_server_port),
        "-c",
        str(settings.lighton_ctx_size),
        "-ngl",
        str(attempt.ngl),
        "-np",
        str(attempt.parallel),
        "-t",
        str(settings.lighton_threads),
        "-tb",
        str(settings.lighton_threads_batch),
        "--jinja",
    ]
    if attempt.flash_attn:
        command.append("--flash-attn")

    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"llama-server not found: {executable}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to start llama-server: {exc}") from exc


def _wait_for_server_ready(
    *,
    process: subprocess.Popen[bytes],
    base_url: str,
    timeout_seconds: float,
) -> tuple[bool, str | None]:
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            detail = _read_stderr_excerpt(process)
            if detail:
                return False, f"exitcode={exit_code}, stderr={detail}"
            return False, f"exitcode={exit_code}"

        if _is_server_healthy(base_url):
            return True, None
        time.sleep(0.2)

    return False, f"health check timed out after {timeout_seconds:.1f}s"


def _is_server_healthy(base_url: str) -> bool:
    health_url = f"{base_url}/health"
    models_url = f"{base_url}/v1/models"
    try:
        with httpx.Client(timeout=1.0, trust_env=False) as client:
            health_response = client.get(health_url)
            if health_response.status_code != 200:
                return False
            models_response = client.get(models_url)
            if models_response.status_code != 200:
                return False
        try:
            payload = models_response.json()
        except ValueError:
            return False
        if not isinstance(payload, dict):
            return False
        return isinstance(payload.get("data"), list)
    except httpx.HTTPError:
        return False


def _read_stderr_excerpt(process: subprocess.Popen[bytes]) -> str:
    if process.stderr is None:
        return ""
    try:
        payload = process.stderr.read()
    except Exception:
        return ""
    if not payload:
        return ""
    text = payload.decode("utf-8", errors="replace").strip()
    if len(text) > 500:
        return text[:500].strip() + "..."
    return text


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=5.0)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=2.0)
        except Exception:
            return
