import re
from typing import cast

import pytest
from fastapi.testclient import TestClient

from ragprep import pymupdf4llm_markdown
from ragprep.web.app import app


def test_root_renders_page() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "PDF" in response.text
    assert "hx-post" in response.text
    assert "/static/htmx.min.js" in response.text

    htmx = client.get("/static/htmx.min.js")
    assert htmx.status_code == 200
    assert "htmx" in htmx.text


def _extract_job_id(html: str) -> str:
    match = re.search(r'data-job-id="([0-9a-f]{32})"', html)
    assert match is not None, f"job id not found in html: {html}"
    return match.group(1)


def _make_pdf_bytes(page_count: int) -> bytes:
    import fitz

    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page()
        page.insert_text((72, 72), f"Hello {i + 1}")
    return cast(bytes, doc.tobytes())


def test_convert_creates_job_and_downloads_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)

    pdf_bytes = _make_pdf_bytes(page_count=2)

    calls: dict[str, int] = {"n": 0}

    def fake_to_markdown(_doc: object) -> str:
        calls["n"] += 1
        return "page1\r\n\r\npage2\r"

    monkeypatch.setattr(pymupdf4llm_markdown.pymupdf4llm, "to_markdown", fake_to_markdown)

    files = {"file": ("test.pdf", pdf_bytes, "application/pdf")}
    response = client.post("/convert", files=files)
    assert response.status_code == 200
    job_id = _extract_job_id(response.text)
    assert f"/jobs/{job_id}/status" in response.text
    assert "<progress" in response.text

    for _ in range(20):
        result = client.get(f"/jobs/{job_id}/result")
        if result.status_code == 200:
            break
        assert result.status_code == 409
    else:
        pytest.fail("job did not complete")

    assert "page1" in result.text
    assert "page2" in result.text
    assert f"/download/{job_id}.md" in result.text
    assert "save_markdown" in result.text

    download = client.get(f"/download/{job_id}.md")
    assert download.status_code == 200
    assert download.text == "page1\n\npage2"
    assert "text/markdown" in download.headers["content-type"]
    assert "test.md" in download.headers["content-disposition"]
    assert calls["n"] == 1

    _ = client.get(f"/jobs/{job_id}/status")
    _ = client.get(f"/download/{job_id}.md")
    assert calls["n"] == 1


def test_bad_pdf_job_reports_error() -> None:
    client = TestClient(app)
    files = {"file": ("bad.pdf", b"not a pdf", "application/pdf")}
    response = client.post("/convert", files=files)
    assert response.status_code == 200
    job_id = _extract_job_id(response.text)

    for _ in range(20):
        status = client.get(f"/jobs/{job_id}/status")
        assert status.status_code == 200
        if "Invalid PDF data" in status.text:
            break
    else:
        pytest.fail("expected invalid pdf error")

    download = client.get(f"/download/{job_id}.md")
    assert download.status_code == 409


def test_convert_rejects_large_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    monkeypatch.setenv("RAGPREP_MAX_UPLOAD_BYTES", "1")
    files = {"file": ("test.pdf", b"00", "application/pdf")}
    response = client.post("/convert", files=files)
    assert response.status_code == 413
    assert "File too large" in response.text
