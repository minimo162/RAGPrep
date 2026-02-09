from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import httpx

from ragprep.config import Settings
from ragprep.ocr.lighton_assets import LightOnAssetPaths, ensure_lighton_assets

_SERVER_LOCK = Lock()
_ACTIVE_PROCESS: subprocess.Popen[bytes] | None = None
_ACTIVE_BASE_URL: str | None = None


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

        assets = ensure_lighton_assets(settings)
        executable = _resolve_llama_server_executable(settings)
        attempts = _build_attempts(settings)

        errors: list[str] = []
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
                errors.append(f"{attempt.name}: {error}")
            else:
                errors.append(f"{attempt.name}: unknown error")

        joined = "; ".join(errors) if errors else "unknown error"
        raise RuntimeError(f"Failed to start llama-server after retries. {joined}")


def prewarm_lighton_server(*, settings: Settings) -> None:
    _ = ensure_server_base_url(settings)


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

    found = shutil.which("llama-server")
    if found:
        return found

    raise RuntimeError(
        "llama-server not found. Install llama.cpp server binary and ensure it is in PATH "
        "or set RAGPREP_LLAMA_SERVER_PATH."
    )


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
            stderr=subprocess.PIPE,
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
    try:
        with httpx.Client(timeout=1.0, trust_env=False) as client:
            response = client.get(health_url)
        return response.status_code == 200
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
