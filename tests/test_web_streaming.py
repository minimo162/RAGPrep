from __future__ import annotations

from typing import Any

import pytest

from ragprep.pipeline import PdfToHtmlProgress, ProgressPhase
from ragprep.web import app as webapp
from ragprep.web.app import JobStatus


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
