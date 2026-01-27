from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from threading import Lock, Semaphore
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from ragprep.config import get_settings
from ragprep.pipeline import PdfToJsonProgress, pdf_to_json

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

logger = logging.getLogger(__name__)


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
    json_output: str | None = None
    error: str | None = None
    partial_markdown: str = ""
    partial_pages: int = 0


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

    def append_partial(self, job_id: str, *, page_index: int, markdown: str) -> Job:
        entry_text = markdown.strip()
        page_block = f"## Page {page_index}\n\n{entry_text}".strip()

        with self._lock:
            existing = self._jobs.get(job_id)
            if existing is None:
                raise KeyError(job_id)

            partial_markdown = existing.partial_markdown.strip()
            if partial_markdown:
                partial_markdown = f"{partial_markdown}\n\n{page_block}".strip()
            else:
                partial_markdown = page_block

            updated = replace(
                existing,
                partial_markdown=partial_markdown,
                partial_pages=max(existing.partial_pages, int(page_index)),
            )
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
            partial_markdown="",
            partial_pages=0,
        )

        def on_progress(update: PdfToJsonProgress) -> None:
            jobs.update(
                job_id,
                phase=update.phase.value,
                processed_pages=update.current,
                total_pages=update.total,
                progress_message=update.message,
            )

        def on_page(page_index: int, markdown: str) -> None:
            try:
                jobs.append_partial(job_id, page_index=page_index, markdown=markdown)
            except Exception:  # noqa: BLE001
                logger.debug("Failed to append partial output for job %s", job_id, exc_info=True)

        try:
            json_output = pdf_to_json(
                pdf_bytes,
                on_progress=on_progress,
                on_page=on_page,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", job_id)
            jobs.update(job_id, status=JobStatus.error, phase="error", error=str(exc))
            return
        jobs.update(
            job_id,
            status=JobStatus.done,
            phase="done",
            json_output=json_output,
            error=None,
        )


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Response:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"json_output": None, "error": None},
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
            {"json_output": None, "error": "Empty upload."},
            status_code=400,
        )
    if len(content) > settings.max_upload_bytes:
        return templates.TemplateResponse(
            request,
            "_result.html",
            {
                "json_output": None,
                "error": f"File too large (>{settings.max_upload_bytes} bytes).",
            },
            status_code=413,
        )
    if not filename.lower().endswith(".pdf"):
        return templates.TemplateResponse(
            request,
            "_result.html",
            {"json_output": None, "error": "Please upload a .pdf file."},
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
            {"json_output": None, "error": "Result not ready yet."},
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


@app.get("/download/{job_id}.json")
def download_json(job_id: str) -> PlainTextResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != JobStatus.done or job.json_output is None:
        raise HTTPException(status_code=409, detail="job not complete")

    download_filename = _download_filename_from_upload(job.filename, suffix="json")
    headers = {"Content-Disposition": f'attachment; filename="{download_filename}"'}
    return PlainTextResponse(
        job.json_output,
        media_type="application/json; charset=utf-8",
        headers=headers,
    )


def _markdown_from_job(job: Job) -> str:
    """
    Build a Markdown artifact for download.

    Prefer the structured LightOnOCR JSON schema (pages[].markdown).
    Fall back to partial_markdown or raw json_output for robustness.
    """
    if job.json_output:
        try:
            payload = json.loads(job.json_output)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            pages = payload.get("pages")
            if isinstance(pages, list):
                parts: list[str] = []
                for page in pages:
                    if not isinstance(page, dict):
                        continue
                    markdown = page.get("markdown")
                    if isinstance(markdown, str):
                        normalized = (
                            markdown.replace("\r\n", "\n")
                            .replace("\r", "\n")
                            .strip()
                        )
                        if normalized:
                            parts.append(normalized)
                if parts:
                    return "\n\n".join(parts)

    if job.partial_markdown:
        return job.partial_markdown.strip()

    return (job.json_output or "").strip()


@app.get("/download/{job_id}.md")
def download_markdown(job_id: str) -> PlainTextResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != JobStatus.done or job.json_output is None:
        raise HTTPException(status_code=409, detail="job not complete")

    markdown = _markdown_from_job(job)
    if markdown:
        markdown = markdown + "\n"

    download_filename = _download_filename_from_upload(job.filename, suffix="md")
    headers = {"Content-Disposition": f'attachment; filename="{download_filename}"'}
    return PlainTextResponse(
        markdown,
        media_type="text/markdown; charset=utf-8",
        headers=headers,
    )


@app.get("/health")
def health() -> PlainTextResponse:
    return PlainTextResponse("ok", media_type="text/plain; charset=utf-8")
