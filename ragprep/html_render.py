from __future__ import annotations

from html import escape

from ragprep.structure_ir import Block, Document, Figure, Heading, Page, Paragraph, Table, Unknown


def render_document_html(document: Document) -> str:
    """
    Render a Document IR to a safe HTML fragment.

    Output is deterministic and escapes all user/model-provided text.
    """

    parts: list[str] = ['<article class="ragprep-document">']
    for page in document.pages:
        parts.append(_render_page(page))
    parts.append("</article>")
    return "\n".join(parts)


def wrap_html_document(fragment_html: str, *, title: str = "RAGPrep") -> str:
    """
    Wrap a fragment in a minimal standalone HTML document.
    """

    safe_title = escape(title, quote=True)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "  <head>",
            '    <meta charset="utf-8" />',
            '    <meta name="viewport" content="width=device-width, initial-scale=1" />',
            f"    <title>{safe_title}</title>",
            "  </head>",
            "  <body>",
            fragment_html,
            "  </body>",
            "</html>",
        ]
    )


def _render_page(page: Page) -> str:
    parts: list[str] = [f'<section data-page="{int(page.page_number)}">']
    for block in page.blocks:
        parts.append(_render_block(block))
    parts.append("</section>")
    return "\n".join(parts)


def _render_block(block: Block) -> str:
    if isinstance(block, Heading):
        level = max(1, min(6, int(block.level)))
        text = escape(block.text, quote=True)
        return f"<h{level}>{text}</h{level}>"
    if isinstance(block, Paragraph):
        text = escape(block.text, quote=True)
        return f"<p>{text}</p>"
    if isinstance(block, Table):
        text = escape(block.text, quote=True)
        return f'<pre data-kind="table">{text}</pre>'
    if isinstance(block, Figure):
        alt = escape(block.alt, quote=True)
        return f"<figure><figcaption>{alt}</figcaption></figure>"
    if isinstance(block, Unknown):
        text = escape(block.text, quote=True)
        return f"<p>{text}</p>"
    raise TypeError(f"Unsupported block type: {type(block)!r}")

