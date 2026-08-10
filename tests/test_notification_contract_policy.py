"""Policy tests for the notification-hygiene contract artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
TRACK = (
    ROOT
    / "conductor"
    / "tracks"
    / "github_notification_hygiene_20260810"
)
CONTRACTS = TRACK / "contracts"


def _schema(name: str) -> dict[str, object]:
    """Load one contract schema from the canonical track directory."""
    return json.loads((CONTRACTS / name).read_text(encoding="utf-8"))


def test_versioned_contract_bundle_is_complete() -> None:
    """Require input, decision, and audit schemas with closed objects."""
    for name in ("notification.schema.json", "decision.schema.json", "audit.schema.json"):
        schema = _schema(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False
        assert schema["properties"]["schema_version"]["const"] == "1.0.0"


def test_decision_contract_is_fail_open() -> None:
    """Only positive automation evidence may authorize a read action."""
    schema = _schema("decision.schema.json")
    actor_classes = schema["$defs"]["actor_class"]["enum"]
    assert actor_classes == [
        "external_human",
        "self",
        "bot",
        "app",
        "workflow",
        "release_automation",
        "unknown",
    ]
    assert schema["properties"]["action"]["enum"] == ["preserve_unread", "mark_read"]
    assert schema["properties"]["confidence"]["enum"] == ["positive", "unknown"]


def test_audit_contract_excludes_notification_content() -> None:
    """Forbid content-bearing or credential-bearing audit fields."""
    schema = _schema("audit.schema.json")
    fields = set(schema["properties"])
    forbidden = {"title", "body", "comment", "content", "token", "authorization"}
    assert fields.isdisjoint(forbidden)
    assert "notification_id" in fields
    assert "rule_id" in fields
    assert "action" in fields


def test_contract_document_pins_safety_invariants() -> None:
    """Keep the human-readable contract aligned with machine schemas."""
    text = (TRACK / "contract.md").read_text(encoding="utf-8")
    required = (
        "Unknown means preserve unread",
        "Reason alone is insufficient",
        "Dry-run cannot write",
        "No third-party repository writes",
        "X-Poll-Interval",
    )
    for invariant in required:
        assert invariant in text
