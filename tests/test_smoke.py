import re
from collections.abc import Callable
from typing import cast

import pytest
from fastapi.testclient import TestClient

import ragprep.web.app as web_app
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


def test_convert_creates_job_and_downloads_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)

    pdf_bytes = _make_pdf_bytes(page_count=2)

    calls: dict[str, int] = {"n": 0}
    expected_fragment = (
        '<section data-page="1">page1</section>\n<section data-page="2">page2</section>'
    )

    def fake_to_html(
        _bytes: bytes,
        *,
        full_document: bool = True,
        on_progress: object | None = None,
        on_page: Callable[[int, str], None] | None = None,
        _page_output_dir: object | None = None,
    ) -> str:
        _ = full_document
        _ = on_progress
        calls["n"] += 1
        if on_page is not None:
            on_page(1, '<section data-page="1">page1</section>')
            on_page(2, '<section data-page="2">page2</section>')
        return expected_fragment

    monkeypatch.setattr(web_app, "pdf_to_html", fake_to_html)

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
    assert f"/download/{job_id}.json" not in result.text
    assert f"/download/{job_id}.html" in result.text
    assert "save_json" not in result.text
    assert "ragprepHandleHtmlDownload" in result.text

    download_html = client.get(f"/download/{job_id}.html")
    assert download_html.status_code == 200
    assert "text/html" in download_html.headers["content-type"]
    assert "test.html" in download_html.headers["content-disposition"]
    assert "page1" in download_html.text
    assert "page2" in download_html.text
    assert "<!doctype html>" in download_html.text.lower()
    assert calls["n"] == 1

    _ = client.get(f"/jobs/{job_id}/status")
    json_download = client.get(f"/download/{job_id}.json")
    assert json_download.status_code == 404
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

    download = client.get(f"/download/{job_id}.json")
    assert download.status_code == 404


def test_convert_rejects_large_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    monkeypatch.setenv("RAGPREP_MAX_UPLOAD_BYTES", "1")
    files = {"file": ("test.pdf", b"00", "application/pdf")}
    response = client.post("/convert", files=files)
    assert response.status_code == 413
    assert "File too large" in response.text
