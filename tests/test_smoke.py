import re

import pytest
from fastapi.testclient import TestClient

from ragprep.web.app import app


def test_root_renders_page() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "PDF" in response.text
    assert "hx-post" in response.text


def _extract_job_id(html: str) -> str:
    match = re.search(r'data-job-id="([0-9a-f]{32})"', html)
    assert match is not None, f"job id not found in html: {html}"
    return match.group(1)


def test_convert_creates_job_and_downloads_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    monkeypatch.setattr("ragprep.web.app.pdf_to_markdown", lambda _pdf_bytes: "hello")
    files = {"file": ("test.pdf", b"%PDF-1.4\n%fake\n", "application/pdf")}
    response = client.post("/convert", files=files)
    assert response.status_code == 200
    job_id = _extract_job_id(response.text)
    assert f"/jobs/{job_id}/status" in response.text

    status = client.get(f"/jobs/{job_id}/status")
    assert status.status_code == 200
    assert job_id in status.text

    result = client.get(f"/jobs/{job_id}/result")
    assert result.status_code == 200
    assert "hello" in result.text
    assert f"/download/{job_id}.md" in result.text

    download = client.get(f"/download/{job_id}.md")
    assert download.status_code == 200
    assert download.text == "hello"
    assert "text/markdown" in download.headers["content-type"]
    assert f"{job_id}.md" in download.headers["content-disposition"]
