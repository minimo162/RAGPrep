from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from ragprep.pipeline import PdfToHtmlProgress, ProgressPhase
from ragprep.web import app as webapp
from ragprep.web.app import JobStatus, app


def test_run_job_appends_partial_output(monkeypatch: pytest.MonkeyPatch) -> None:
    job = webapp.jobs.create(filename="sample.pdf")

    def _fake_pdf_to_html(
        _pdf_bytes: bytes,
        *,
        full_document: bool = True,
        on_progress: Any = None,
        on_page: Any = None,
        _page_output_dir: Any = None,
    ) -> str:
        _ = full_document
        if on_progress is not None:
            on_progress(
                PdfToHtmlProgress(
                    phase=ProgressPhase.rendering,
                    current=0,
                    total=2,
                    message="converting",
                )
            )
        if on_page is not None:
            on_page(1, '<section data-page="1"><h1>PAGE1</h1></section>')
            on_page(2, '<section data-page="2"><p>PAGE2</p></section>')
        if on_progress is not None:
            on_progress(
                PdfToHtmlProgress(
                    phase=ProgressPhase.done,
                    current=2,
                    total=2,
                    message="done",
                )
            )
        return "<article>DONE</article>"

    monkeypatch.setattr(webapp, "pdf_to_html", _fake_pdf_to_html)

    webapp._run_job(job.id, b"%PDF")
    updated = webapp.jobs.get(job.id)

    assert updated is not None
    assert updated.status == JobStatus.done
    assert updated.partial_pages == 2
    assert updated.html_output == "<article>DONE</article>"
    assert 'data-page="1"' in updated.partial_html
    assert "PAGE2" in updated.partial_html


def test_run_job_sets_error_on_layout_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    job = webapp.jobs.create(filename="sample.pdf")

    def _fake_pdf_to_html(
        _pdf_bytes: bytes,
        *,
        full_document: bool = True,
        on_progress: Any = None,
        on_page: Any = None,
        _page_output_dir: Any = None,
    ) -> str:
        _ = _pdf_bytes, full_document, on_progress, on_page, _page_output_dir
        raise RuntimeError(
            "Layout analysis request timed out. base_url='http://127.0.0.1:8080'. "
            "Ensure the layout server is running and reachable."
        )

    monkeypatch.setattr(webapp, "pdf_to_html", _fake_pdf_to_html)

    webapp._run_job(job.id, b"%PDF")
    updated = webapp.jobs.get(job.id)

    assert updated is not None
    assert updated.status == JobStatus.error
    assert updated.phase == "error"
    assert updated.error is not None
    assert "Layout analysis request timed out" in updated.error


def test_job_status_returns_204_when_version_matches() -> None:
    client = TestClient(app)
    job = webapp.jobs.create(filename="sample.pdf")
    response = client.get(f"/jobs/{job.id}/status", headers={"If-None-Match": str(job.version)})
    assert response.status_code == 204


def test_partial_output_keeps_all_processed_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    # Legacy env should not cap the output anymore.
    monkeypatch.setenv("RAGPREP_WEB_PARTIAL_PREVIEW_PAGES", "1")
    job = webapp.jobs.create(filename="sample.pdf")
    webapp.jobs.append_partial(job.id, page_index=1, html='<section data-page="1">PAGE1</section>')
    webapp.jobs.append_partial(job.id, page_index=2, html='<section data-page="2">PAGE2</section>')
    webapp.jobs.append_partial(job.id, page_index=3, html='<section data-page="3">PAGE3</section>')

    updated = webapp.jobs.get(job.id)
    assert updated is not None
    assert updated.partial_preview_pages == 3
    assert 'data-page="1"' in updated.partial_html
    assert "PAGE2" in updated.partial_html
    assert "PAGE3" in updated.partial_html


def test_should_prewarm_on_startup_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPREP_WEB_PREWARM_ON_STARTUP", "0")
    assert webapp._should_prewarm_on_startup() is False

    monkeypatch.setenv("RAGPREP_WEB_PREWARM_ON_STARTUP", "1")
    assert webapp._should_prewarm_on_startup() is True


def test_ensure_startup_prewarm_started_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake_prewarm() -> None:
        calls.append("called")

    class _ImmediateThread:
        def __init__(self, *, target: Any, daemon: bool, name: str) -> None:
            _ = daemon, name
            self._target = target

        def start(self) -> None:
            self._target()

    monkeypatch.setattr(webapp, "_prewarm_layout_backend", _fake_prewarm)
    monkeypatch.setattr(webapp, "Thread", _ImmediateThread)

    webapp._prewarm_started = False
    webapp._ensure_startup_prewarm_started()
    webapp._ensure_startup_prewarm_started()

    assert calls == ["called"]
