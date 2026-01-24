from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from threading import Event, Lock, Semaphore, Thread
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from ragprep.config import get_settings
from ragprep.pipeline import PdfToMarkdownProgress, pdf_to_markdown

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> Any:
    _start_warmup_if_enabled()
    yield


app = FastAPI(lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

logger = logging.getLogger(__name__)

ENV_WARMUP_ON_START = "RAGPREP_WARMUP_ON_START"


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


_warmup_lock = Lock()
_warmup_started = False
_warmup_done = Event()
_warmup_error: str | None = None


def _get_lightonocr_backend_for_warmup() -> str:
    raw = os.getenv("LIGHTONOCR_BACKEND")
    if raw is None or not raw.strip():
        return "cli"

    value = raw.strip().lower()
    if value in {"cli"}:
        return "cli"
    if value in {"python", "py"}:
        return "python"

    return "invalid"


def _warmup_impl() -> None:
    backend = _get_lightonocr_backend_for_warmup()
    if backend != "python":
        return

    from ragprep.ocr import lightonocr

    settings = lightonocr.get_settings()

    from ragprep.ocr.llamacpp_runtime import _get_runtime_cached

    _get_runtime_cached(
        settings.gguf_model_path,
        settings.gguf_mmproj_path,
        settings.llama_n_ctx,
        settings.llama_n_threads,
        settings.llama_n_gpu_layers,
    )


def _run_warmup_thread() -> None:
    global _warmup_error
    try:
        _warmup_impl()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Warmup failed")
        _warmup_error = str(exc)
    finally:
        _warmup_done.set()


def _start_warmup_if_enabled() -> None:
    global _warmup_started

    if not _env_truthy(ENV_WARMUP_ON_START):
        return

    with _warmup_lock:
        if _warmup_started:
            return
        _warmup_started = True

        thread = Thread(target=_run_warmup_thread, name="ragprep_warmup", daemon=True)
        thread.start()


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"


@dataclass(frozen=True)
class Job:
    id: str
    filename: str
    status: JobStatus
    phase: str | None = None
    processed_pages: int = 0
    total_pages: int = 0
    progress_message: str | None = None
    markdown: str | None = None
    error: str | None = None


class JobStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._jobs: dict[str, Job] = {}

    def create(self, *, filename: str) -> Job:
        job = Job(id=uuid.uuid4().hex, filename=filename, status=JobStatus.queued)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: Any) -> Job:
        with self._lock:
            existing = self._jobs.get(job_id)
            if existing is None:
                raise KeyError(job_id)
            updated = replace(existing, **changes)
            self._jobs[job_id] = updated
            return updated


jobs = JobStore()


_job_semaphore_lock = Lock()
_job_semaphore: Semaphore | None = None
_job_semaphore_size: int | None = None


def _get_job_semaphore() -> Semaphore:
    global _job_semaphore, _job_semaphore_size

    size = get_settings().max_concurrency
    with _job_semaphore_lock:
        if _job_semaphore is None or _job_semaphore_size != size:
            _job_semaphore = Semaphore(size)
            _job_semaphore_size = size
        return _job_semaphore


def _run_job(job_id: str, pdf_bytes: bytes) -> None:
    semaphore = _get_job_semaphore()
    with semaphore:
        jobs.update(
            job_id,
            status=JobStatus.running,
            error=None,
            phase="starting",
            processed_pages=0,
            total_pages=0,
            progress_message=None,
        )

        _start_warmup_if_enabled()
        if _env_truthy(ENV_WARMUP_ON_START) and not _warmup_done.is_set():
            jobs.update(
                job_id,
                phase="warming_up",
                progress_message="Warming up OCR runtime...",
            )
            _warmup_done.wait()
            jobs.update(job_id, phase="starting", progress_message=None)

        def on_progress(update: PdfToMarkdownProgress) -> None:
            jobs.update(
                job_id,
                phase=update.phase.value,
                processed_pages=update.current,
                total_pages=update.total,
                progress_message=update.message,
            )

        try:
            markdown = pdf_to_markdown(pdf_bytes, on_progress=on_progress)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", job_id)
            jobs.update(job_id, status=JobStatus.error, phase="error", error=str(exc))
            return
        jobs.update(job_id, status=JobStatus.done, phase="done", markdown=markdown, error=None)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Response:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"markdown": None, "error": None},
    )


@app.post("/convert", response_class=HTMLResponse)
async def convert(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),  # noqa: B008
) -> Response:
    settings = get_settings()
    filename = file.filename or "upload"
    content = await file.read()

    if not content:
        return templates.TemplateResponse(
            request,
            "_result.html",
            {"markdown": None, "error": "Empty upload."},
            status_code=400,
        )
    if len(content) > settings.max_upload_bytes:
        return templates.TemplateResponse(
            request,
            "_result.html",
            {
                "markdown": None,
                "error": f"File too large (>{settings.max_upload_bytes} bytes).",
            },
            status_code=413,
        )
    if not filename.lower().endswith(".pdf"):
        return templates.TemplateResponse(
            request,
            "_result.html",
            {"markdown": None, "error": "Please upload a .pdf file."},
            status_code=400,
        )

    job = jobs.create(filename=filename)
    background_tasks.add_task(_run_job, job.id, content)

    return templates.TemplateResponse(
        request,
        "_job_status.html",
        {"job": job},
    )


@app.get("/jobs/{job_id}/status", response_class=HTMLResponse)
def job_status(request: Request, job_id: str) -> Response:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return templates.TemplateResponse(request, "_job_status.html", {"job": job})


@app.get("/jobs/{job_id}/result", response_class=HTMLResponse)
def job_result(request: Request, job_id: str) -> Response:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != JobStatus.done:
        return templates.TemplateResponse(
            request,
            "_result.html",
            {"markdown": None, "error": "Result not ready yet."},
            status_code=409,
        )
    return templates.TemplateResponse(request, "_job_result.html", {"job": job})


@app.get("/download/{job_id}.md")
def download_markdown(job_id: str) -> PlainTextResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != JobStatus.done or job.markdown is None:
        raise HTTPException(status_code=409, detail="job not complete")

    headers = {"Content-Disposition": f'attachment; filename="{job_id}.md"'}
    return PlainTextResponse(
        job.markdown,
        media_type="text/markdown; charset=utf-8",
        headers=headers,
    )
