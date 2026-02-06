from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from threading import Lock, Semaphore, Thread
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from ragprep.config import get_settings
from ragprep.html_render import wrap_html_document
from ragprep.pipeline import PdfToHtmlProgress, pdf_to_html

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

logger = logging.getLogger(__name__)

_PARTIAL_PAGE_SEPARATOR = "\n<!-- ragprep:partial-page -->\n"
ENV_WEB_PREWARM_ON_STARTUP = "RAGPREP_WEB_PREWARM_ON_STARTUP"
DEFAULT_WEB_PREWARM_ON_STARTUP = True


def _get_bool_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _should_prewarm_on_startup() -> bool:
    return _get_bool_env(ENV_WEB_PREWARM_ON_STARTUP, default=DEFAULT_WEB_PREWARM_ON_STARTUP)


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
    version: int = 0
    phase: str | None = None
    processed_pages: int = 0
    total_pages: int = 0
    progress_message: str | None = None
    html_output: str | None = None
    error: str | None = None
    partial_html: str = ""
    partial_pages: int = 0
    partial_preview_pages: int = 0


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
            next_version = int(existing.version) + 1
            updated = replace(existing, version=next_version, **changes)
            self._jobs[job_id] = updated
            return updated

    def append_partial(self, job_id: str, *, page_index: int, html: str) -> Job:
        page_block = html.strip()

        with self._lock:
            existing = self._jobs.get(job_id)
            if existing is None:
                raise KeyError(job_id)

            existing_html = existing.partial_html.strip()
            blocks = (
                [b for b in existing_html.split(_PARTIAL_PAGE_SEPARATOR) if b.strip()]
                if existing_html
                else []
            )
            blocks.append(page_block)

            partial_html = _PARTIAL_PAGE_SEPARATOR.join(blocks).strip()
            partial_pages = max(existing.partial_pages, int(page_index))

            updated = replace(
                existing,
                version=int(existing.version) + 1,
                partial_html=partial_html,
                partial_pages=partial_pages,
                # Keep this for template compatibility; now reflects total accumulated pages.
                partial_preview_pages=partial_pages,
            )
            self._jobs[job_id] = updated
            return updated

    def has_active(self) -> bool:
        with self._lock:
            return any(
                j.status in {JobStatus.queued, JobStatus.running}
                for j in self._jobs.values()
            )


jobs = JobStore()


_job_semaphore_lock = Lock()
_job_semaphore: Semaphore | None = None
_job_semaphore_size: int | None = None

_prewarm_lock = Lock()
_prewarm_started = False
_prewarm_state_lock = Lock()
_prewarm_enabled = False
_prewarm_in_progress = False
_prewarm_error: str | None = None
_PREWARM_UNCHANGED = object()


def _get_job_semaphore() -> Semaphore:
    global _job_semaphore, _job_semaphore_size

    size = get_settings().max_concurrency
    with _job_semaphore_lock:
        if _job_semaphore is None or _job_semaphore_size != size:
            _job_semaphore = Semaphore(size)
            _job_semaphore_size = size
        return _job_semaphore


def _set_prewarm_state(
    *,
    enabled: bool | None = None,
    in_progress: bool | None = None,
    error: str | None | object = _PREWARM_UNCHANGED,
) -> None:
    global _prewarm_enabled, _prewarm_in_progress, _prewarm_error

    with _prewarm_state_lock:
        if enabled is not None:
            _prewarm_enabled = enabled
        if in_progress is not None:
            _prewarm_in_progress = in_progress
        if error is not _PREWARM_UNCHANGED:
            _prewarm_error = error if isinstance(error, str) else None


def _get_prewarm_state() -> dict[str, object]:
    with _prewarm_state_lock:
        return {
            "prewarm_enabled": _prewarm_enabled,
            "prewarm_in_progress": _prewarm_in_progress,
            "prewarm_error": _prewarm_error,
        }


def _prewarm_layout_backend() -> str | None:
    settings = get_settings()
    try:
        from ragprep.layout.glm_doclayout import prewarm_layout_backend

        prewarm_layout_backend(settings=settings)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Layout prewarm failed: %s", exc)
        return str(exc)
    return None


def _run_prewarm() -> None:
    error = _prewarm_layout_backend()
    _set_prewarm_state(in_progress=False, error=error)


def _ensure_startup_prewarm_started() -> None:
    global _prewarm_started

    with _prewarm_lock:
        if _prewarm_started:
            return
        _prewarm_started = True
    Thread(target=_run_prewarm, daemon=True, name="ragprep-prewarm").start()


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
            partial_html="",
            partial_pages=0,
            partial_preview_pages=0,
        )

        def on_progress(update: PdfToHtmlProgress) -> None:
            jobs.update(
                job_id,
                phase=update.phase.value,
                processed_pages=update.current,
                total_pages=update.total,
                progress_message=update.message,
            )

        def on_page(page_index: int, html: str) -> None:
            try:
                jobs.append_partial(job_id, page_index=page_index, html=html)
            except Exception:  # noqa: BLE001
                logger.debug("Failed to append partial output for job %s", job_id, exc_info=True)

        try:
            html_output = pdf_to_html(
                pdf_bytes,
                full_document=False,
                on_progress=on_progress,
                on_page=on_page,
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            expected_errors = (
                "Failed to reach GLM-OCR server",
                "GLM-OCR request timed out",
                "GLM-OCR server is not reachable",
                "Failed to reach layout server",
                "Layout analysis request timed out",
                "Failed to load GLM-OCR processor via Transformers",
                "Failed to load GLM-OCR model via Transformers",
                "Transformers backend selected, but required packages are missing",
                "argument of type 'NoneType' is not iterable",
            )
            if any(token in message for token in expected_errors):
                logger.warning("Job %s failed: %s", job_id, message)
            else:
                logger.exception("Job %s failed", job_id)
            jobs.update(job_id, status=JobStatus.error, phase="error", error=str(exc))
            return
        jobs.update(
            job_id,
            status=JobStatus.done,
            phase="done",
            html_output=html_output,
            error=None,
        )


def _start_job(job_id: str, pdf_bytes: bytes) -> None:
    # Avoid tying long-running sync work to the request lifecycle/threadpool.
    Thread(target=_run_job, args=(job_id, pdf_bytes), daemon=True).start()


@app.on_event("startup")
def startup() -> None:
    enabled = _should_prewarm_on_startup()
    if enabled:
        _set_prewarm_state(enabled=True, in_progress=True, error=None)
        _ensure_startup_prewarm_started()
        return
    _set_prewarm_state(enabled=False, in_progress=False, error=None)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Response:
    ui_state = _get_prewarm_state()
    ui_state["has_active_job"] = jobs.has_active()
    return templates.TemplateResponse(
        request,
        "index.html",
        {"html_output": None, "error": None, "ui_state": ui_state},
    )


@app.get("/ui/state")
def ui_state() -> JSONResponse:
    state = _get_prewarm_state()
    state["has_active_job"] = jobs.has_active()
    state["can_convert"] = (not bool(state["prewarm_in_progress"])) and (
        not bool(state["has_active_job"])
    )
    return JSONResponse(state)


@app.post("/convert", response_class=HTMLResponse)
async def convert(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
) -> Response:
    settings = get_settings()
    filename = file.filename or "upload"
    content = await file.read()
    prewarm_state = _get_prewarm_state()

    if bool(prewarm_state["prewarm_in_progress"]):
        return templates.TemplateResponse(
            request,
            "_result.html",
            {"html_output": None, "error": "Prewarm in progress. Please wait for completion."},
            status_code=409,
        )

    if jobs.has_active():
        return templates.TemplateResponse(
            request,
            "_result.html",
            {"html_output": None, "error": "Conversion is already running. Please wait."},
            status_code=409,
        )

    if not content:
        return templates.TemplateResponse(
            request,
            "_result.html",
            {"html_output": None, "error": "Empty upload."},
            status_code=400,
        )
    if len(content) > settings.max_upload_bytes:
        return templates.TemplateResponse(
            request,
            "_result.html",
            {
                "html_output": None,
                "error": f"File too large (>{settings.max_upload_bytes} bytes).",
            },
            status_code=413,
        )
    if not filename.lower().endswith(".pdf"):
        return templates.TemplateResponse(
            request,
            "_result.html",
            {"html_output": None, "error": "Please upload a .pdf file."},
            status_code=400,
        )

    job = jobs.create(filename=filename)
    _start_job(job.id, content)

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
    if_none_match = request.headers.get("if-none-match")
    if if_none_match is not None and if_none_match.strip() == str(job.version):
        return Response(status_code=204)
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
            {"html_output": None, "error": "Result not ready yet."},
            status_code=409,
        )
    return templates.TemplateResponse(request, "_job_result.html", {"job": job})


def _safe_download_stem(upload_filename: str) -> str:
    name = Path(upload_filename).name
    name = name.replace("\r", "").replace("\n", "")
    stem = Path(name).stem.replace('"', "")
    if not stem or stem in {".", ".."}:
        stem = "download"
    return stem


def _download_filename_from_upload(upload_filename: str, *, suffix: str) -> str:
    stem = _safe_download_stem(upload_filename)
    return f"{stem}.{suffix}"

def _html_fragment_from_job(job: Job) -> str:
    if job.html_output:
        return job.html_output.strip()
    if job.partial_html:
        return job.partial_html.strip()
    return ""


@app.get("/download/{job_id}.html")
def download_html(job_id: str) -> HTMLResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != JobStatus.done or job.html_output is None:
        raise HTTPException(status_code=409, detail="job not complete")

    fragment = _html_fragment_from_job(job)
    html = wrap_html_document(fragment, title=_safe_download_stem(job.filename))

    download_filename = _download_filename_from_upload(job.filename, suffix="html")
    headers = {"Content-Disposition": f'attachment; filename="{download_filename}"'}
    return HTMLResponse(html, media_type="text/html; charset=utf-8", headers=headers)


@app.get("/health")
def health() -> PlainTextResponse:
    return PlainTextResponse("ok", media_type="text/plain; charset=utf-8")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)
