from __future__ import annotations

from typing import Any

import pytest

from ragprep.pipeline import PdfToJsonProgress, ProgressPhase
from ragprep.web import app as webapp
from ragprep.web.app import JobStatus


def test_run_job_appends_partial_output(monkeypatch: pytest.MonkeyPatch) -> None:
    job = webapp.jobs.create(filename="sample.pdf")

    def _fake_pdf_to_json(
        _pdf_bytes: bytes,
        *,
        on_progress: Any = None,
        on_page: Any = None,
        _page_output_dir: Any = None,
    ) -> str:
        if on_progress is not None:
            on_progress(
                PdfToJsonProgress(
                    phase=ProgressPhase.rendering,
                    current=0,
                    total=2,
                    message="converting",
                )
            )
        if on_page is not None:
            on_page(1, "PAGE1")
            on_page(2, "PAGE2")
        if on_progress is not None:
            on_progress(
                PdfToJsonProgress(
                    phase=ProgressPhase.done,
                    current=2,
                    total=2,
                    message="done",
                )
            )
        return '{"ok": true}'

    monkeypatch.setattr(webapp, "pdf_to_json", _fake_pdf_to_json)

    webapp._run_job(job.id, b"%PDF")
    updated = webapp.jobs.get(job.id)

    assert updated is not None
    assert updated.status == JobStatus.done
    assert updated.partial_pages == 2
    assert "## Page 1" in updated.partial_markdown
    assert "PAGE2" in updated.partial_markdown
