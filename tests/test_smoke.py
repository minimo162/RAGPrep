from fastapi.testclient import TestClient

from ragprep.web.app import PLACEHOLDER_MARKDOWN, app


def test_root_renders_page() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "PDF" in response.text
    assert "hx-post" in response.text


def test_convert_returns_placeholder_markdown() -> None:
    client = TestClient(app)
    files = {"file": ("test.pdf", b"%PDF-1.4\n%fake\n", "application/pdf")}
    response = client.post("/convert", files=files)
    assert response.status_code == 200
    assert PLACEHOLDER_MARKDOWN in response.text
