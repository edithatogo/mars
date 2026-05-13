"""Tests for the roadmap closure audit note."""

from __future__ import annotations

from pathlib import Path


def test_roadmap_closure_audit_mentions_deferred_items() -> None:
    """The closure audit note should keep deferred work explicit."""
    content = (
        Path(__file__).resolve().parents[1] / "docs" / "roadmap_closure_audit.md"
    ).read_text()

    assert "Deferred" in content or "deferred" in content
    assert "external" in content.lower()
    assert "Real GPU/TPU/FPGA/ASIC kernels" in content
