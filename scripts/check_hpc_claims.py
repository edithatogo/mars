#!/usr/bin/env python3
"""Validate HPC claim text and upstream-lane metadata before upstream submissions."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ClaimRule:
    """Rule describing a claim token and the minimum allowed HPC level."""

    pattern: re.Pattern[str]
    required_level: int
    message: str


CLAIM_LEVELS = {"H0": 0, "H1": 1, "H2": 2, "H3": 3, "H4": 4}
IMPLEMENTED_MAX_LEVEL = CLAIM_LEVELS["H1"]

CLAIM_RULES = [
    ClaimRule(re.compile(r"\b(GPU|CUDA|ROCm|Metal|TPU)\b", re.IGNORECASE), 3, "accelerator"),
    ClaimRule(re.compile(r"\baccelerator\b", re.IGNORECASE), 3, "accelerator"),
    ClaimRule(re.compile(r"\bMPI\b", re.IGNORECASE), 4, "distributed execution"),
    ClaimRule(re.compile(r"\bdistributed\b", re.IGNORECASE), 4, "distributed execution"),
    ClaimRule(re.compile(r"\bmulti-?node\b", re.IGNORECASE), 4, "distributed execution"),
    ClaimRule(re.compile(r"\bmulti[- ]?worker\b", re.IGNORECASE), 4, "distributed execution"),
]

ALLOWED_CONTEXT_PATTERNS = [
    re.compile(r"\bnon-?goals?\b", re.IGNORECASE),
    re.compile(r"\bnot\s+(yet|currently|ready|supported)\b", re.IGNORECASE),
    re.compile(r"\bmust\s+not\b", re.IGNORECASE),
    re.compile(r"\bhpc[- ]+contract\b", re.IGNORECASE),
    re.compile(r"\bH[0-4]\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\b", re.IGNORECASE),
    re.compile(r"\bplan\b", re.IGNORECASE),
    re.compile(r"\bupstream[- ]+submission\b", re.IGNORECASE),
    re.compile(r"\bphase\b", re.IGNORECASE),
    re.compile(r"\bcurrently\b", re.IGNORECASE),
]

ACCELERATOR_PENDING_CONTEXT_PATTERNS = [
    re.compile(r"\bnot\s+(yet|currently|supported)\b", re.IGNORECASE),
    re.compile(r"\bnon-?goals?\b", re.IGNORECASE),
    re.compile(r"\bmust\s+not\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\b", re.IGNORECASE),
    re.compile(r"\bupcoming\b", re.IGNORECASE),
]

HPC_TARGET_FILES = [
    ROOT / "docs" / "community_submission_readiness.md",
    ROOT / "docs" / "hpc_contracts.md",
    ROOT / "docs" / "hpc_parallel_execution_guide.md",
    ROOT / "docs" / "package_release_paths.md",
    ROOT / "docs" / "release_checklist.md",
    ROOT / "docs" / "release_inventory.md",
    ROOT / "docs" / "publication_handoff.md",
    ROOT / "docs" / "hpc_claim_review_checklist.md",
    ROOT / "packaging" / "spack" / "README.md",
    ROOT / "packaging" / "spack" / "package.py",
    ROOT / "packaging" / "easybuild" / "README.md",
    ROOT / "packaging" / "easybuild" / "pymars-0.1.0.eb",
    ROOT / "packaging" / "conda-forge" / "README.md",
    ROOT / "packaging" / "conda-forge" / "recipe" / "meta.yaml",
]

PLACEHOLDER_PATTERNS = [
    re.compile(r"\bplaceholder\b", re.IGNORECASE),
    re.compile(r"\bTODO\b", re.IGNORECASE),
    re.compile(r"\bFIXME\b", re.IGNORECASE),
    re.compile(r"\bTBD\b", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the HPC claim checker."""
    parser = argparse.ArgumentParser(description="Validate HPC claim wording and lane placeholders.")
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        metavar="PATH",
        help="Additional file(s) to scan for claim violations.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if any optional reference file is missing.",
    )
    return parser.parse_args()


def _has_allowed_context(line: str) -> bool:
    return any(pattern.search(line) for pattern in ALLOWED_CONTEXT_PATTERNS)


def _has_accelerator_pending_context(line: str) -> bool:
    return any(pattern.search(line) for pattern in ACCELERATOR_PENDING_CONTEXT_PATTERNS)


def _line_has_placeholder(line: str) -> bool:
    return any(pattern.search(line) for pattern in PLACEHOLDER_PATTERNS)


def check_claims(path: Path, errors: list[str]) -> None:
    """Append unsupported HPC claim wording found in a file."""
    for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = raw.strip()
        for rule in CLAIM_RULES:
            if not rule.pattern.search(line):
                continue
            if _has_allowed_context(line):
                continue
            if (
                rule.pattern.pattern == r"\baccelerator\b"
                and _has_accelerator_pending_context(line)
            ):
                continue
            if rule.required_level <= IMPLEMENTED_MAX_LEVEL:
                continue
            errors.append(
                f"{path}:{line_no}: unsupported claim '{rule.message}' may imply H{rule.required_level}+ before proof exists."
            )


def check_placeholders(path: Path, errors: list[str], strict: bool) -> None:
    """Append placeholder-token violations found in a file."""
    if not strict and path.name in {"package.py", "pymars-0.1.0.eb", "meta.yaml"}:
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if any(pattern.search(raw) for pattern in PLACEHOLDER_PATTERNS):
            errors.append(f"{path}:{line_no}: placeholder token remains")


def main() -> int:
    """Validate the tracked HPC claim and placeholder files."""
    args = parse_args()
    errors: list[str] = []
    targets = HPC_TARGET_FILES + [Path(path) for path in args.path]
    missing: list[Path] = []
    seen: set[Path] = set()
    for path in targets:
        if not path.exists():
            if path not in seen:
                missing.append(path)
                seen.add(path)
            continue
        check_claims(path, errors)
        check_placeholders(path, errors, args.strict)
        seen.add(path)
    if missing:
        missing_message = ", ".join(str(item) for item in sorted({str(path) for path in missing}))
        if args.strict:
            errors.append(f"missing required target file(s): {missing_message}")
        else:
            print(f"check_hpc_claims.py: skipping missing files (use --strict): {missing_message}", file=sys.stderr)

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("hpc claim-check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
