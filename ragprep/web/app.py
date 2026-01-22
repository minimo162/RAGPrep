from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
PLACEHOLDER_MARKDOWN = "TODO: OCR pipeline not wired yet"

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Response:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"markdown": None, "error": None},
    )


@app.post("/convert", response_class=HTMLResponse)
async def convert(request: Request, file: UploadFile = File(...)) -> Response:  # noqa: B008
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

    return templates.TemplateResponse(
        request,
        "_result.html",
        {"markdown": PLACEHOLDER_MARKDOWN, "error": None},
    )
