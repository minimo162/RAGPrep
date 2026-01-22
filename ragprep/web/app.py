from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from ragprep.pipeline import pdf_to_markdown

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI()


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


def _run_job(job_id: str, pdf_bytes: bytes) -> None:
    jobs.update(job_id, status=JobStatus.running, error=None)
    try:
        markdown = pdf_to_markdown(pdf_bytes)
    except Exception as exc:  # noqa: BLE001
        jobs.update(job_id, status=JobStatus.error, error=str(exc))
        return
    jobs.update(job_id, status=JobStatus.done, markdown=markdown, error=None)


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
    filename = file.filename or "upload"
    content = await file.read()

    if not content:
        return templates.TemplateResponse(
            request,
            "_result.html",
            {"markdown": None, "error": "Empty upload."},
            status_code=400,
        )
    if len(content) > MAX_UPLOAD_BYTES:
        return templates.TemplateResponse(
            request,
            "_result.html",
            {
                "markdown": None,
                "error": f"File too large (>{MAX_UPLOAD_BYTES} bytes).",
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
