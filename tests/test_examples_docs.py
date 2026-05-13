"""Tests for the example and notebook documentation surface."""

from __future__ import annotations

import json
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs" / "examples"


def test_examples_notebook_is_valid_json() -> None:
    """The canonical example notebook should be structurally valid."""
    notebook_path = DOCS_DIR / "python_workflows.ipynb"
    payload = json.loads(notebook_path.read_text())

    assert payload["nbformat"] == 4
    assert payload["nbformat_minor"] == 5
    assert len(payload["cells"]) >= 2
    assert payload["cells"][0]["cell_type"] == "markdown"


def test_examples_pages_exist() -> None:
    """The docs examples index should point at concrete example pages."""
    expected = {
        "index.md",
        "python.md",
        "python_workflows.ipynb",
        "csharp_workflows.ipynb",
        "go_workflows.ipynb",
        "julia_workflows.ipynb",
        "r_workflows.ipynb",
        "rust_workflows.ipynb",
        "rust.md",
        "go.md",
        "julia.md",
        "r.md",
        "csharp.md",
        "typescript.md",
        "typescript_workflows.ipynb",
    }

    assert expected.issubset({path.name for path in DOCS_DIR.iterdir()})
