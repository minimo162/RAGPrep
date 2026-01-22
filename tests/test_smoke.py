from fastapi.testclient import TestClient

from ragprep.web.app import app


def test_root_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
