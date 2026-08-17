"""Safe packaged-template renderer for Living Session constellations."""

from __future__ import annotations

import html
import json
import os
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .constellation import ConstellationGraph


def render_constellation_html(
    graph: "ConstellationGraph",
    *,
    redact_notes: bool = False,
) -> str:
    """Render one self-contained offline HTML navigator from a packaged template."""

    payload = graph.to_dict(redact_notes=redact_notes)
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    template = (
        files("pysbagen")
        .joinpath("data")
        .joinpath("constellation_template.html")
        .read_text(encoding="utf-8")
    )
    title = "PySbagen Living Session Constellation"
    return (
        template.replace("__TITLE__", html.escape(title))
        .replace("__GRAPH_SHORT__", payload["graph_sha256"][:16])
        .replace("__REDACTED__", str(redact_notes).lower())
        .replace("__DATA_JSON__", data_json)
    )


def write_constellation_html(
    graph: "ConstellationGraph",
    destination: str | Path,
    *,
    redact_notes: bool = False,
) -> Path:
    """Atomically write a self-contained HTML constellation snapshot."""

    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            render_constellation_html(graph, redact_notes=redact_notes),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path.resolve()


def install_constellation_renderer() -> None:
    """Replace the embedded prototype renderer with the packaged-template implementation."""

    from . import constellation

    constellation.render_constellation_html = render_constellation_html
    constellation.write_constellation_html = write_constellation_html
