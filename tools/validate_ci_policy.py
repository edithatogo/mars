"""Validate repository CI/CD policy without executing workflow code."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

SHA_PIN = re.compile(r"uses:\s+([^\s#]+)@([^\s#]+)")
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
LOCAL_SPECIFIER = re.compile(r"^(?:file:|link:|/|[A-Za-z]:[\\/])")
REQUIRED_RENOVATE_MANAGERS = {
    "cargo",
    "github-actions",
    "git-submodules",
    "gomod",
    "npm",
    "pep621",
}
REQUIRED_CONTROL_FIELDS = {
    "id",
    "title",
    "tier",
    "trigger",
    "commands",
    "workflows",
    "evidence",
    "failurePolicy",
    "remediation",
    "state",
}


@dataclass(frozen=True, order=True)
class PolicyViolation:
    """One deterministic CI policy violation."""

    code: str
    path: str
    message: str


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _top_level_value(text: str, key: str) -> str | None:
    """Return one top-level YAML scalar or indented mapping as text."""
    lines = text.splitlines()
    prefix = f"{key}:"
    for index, line in enumerate(lines):
        if not line.startswith(prefix):
            continue
        inline = line.removeprefix(prefix).strip()
        if inline:
            return inline
        body: list[str] = []
        for candidate in lines[index + 1 :]:
            if candidate and not candidate[0].isspace():
                break
            body.append(candidate)
        return "\n".join(body)
    return None


def _upload_blocks(text: str) -> list[str]:
    """Return the YAML step block for every upload-artifact invocation."""
    matches = list(
        re.finditer(
            r"(?m)^(?P<indent>\s*)-\s+uses:\s+actions/upload-artifact@[^\s#]+.*$",
            text,
        )
    )
    blocks: list[str] = []
    for match in matches:
        indent = match.group("indent")
        next_step = re.search(rf"(?m)^{re.escape(indent)}-\s+", text[match.end() :])
        end = match.end() + next_step.start() if next_step else len(text)
        blocks.append(text[match.start() : end])
    return blocks


def _workflow_violations(root: Path) -> list[PolicyViolation]:
    violations: list[PolicyViolation] = []
    workflow_dir = root / ".github" / "workflows"
    workflows = sorted((*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")))
    for path in workflows:
        rel = _relative(path, root)
        text = path.read_text(encoding="utf-8")
        if not re.search(r"(?m)^permissions:(?:\s*\{\s*\}|\s*)$", text):
            violations.append(
                PolicyViolation(
                    "workflow-permissions-missing",
                    rel,
                    "Workflow has no explicit top-level permissions declaration",
                )
            )
        concurrency = _top_level_value(text, "concurrency")
        if concurrency is None or not (
            (concurrency and not concurrency[0].isspace())
            or re.search(r"(?m)^\s+group:\s*\S+", concurrency)
        ):
            violations.append(
                PolicyViolation(
                    "workflow-concurrency-missing",
                    rel,
                    "Workflow has no explicit concurrency group",
                )
            )
        if any(
            not re.search(r"(?m)^\s+retention-days:\s*\d+\s*$", block)
            for block in _upload_blocks(text)
        ):
            violations.append(
                PolicyViolation(
                    "artifact-retention-missing",
                    rel,
                    "Artifact upload has no explicit retention-days",
                )
            )
        for match in SHA_PIN.finditer(text):
            action, reference = match.groups()
            if not FULL_SHA.fullmatch(reference):
                violations.append(
                    PolicyViolation(
                        "action-not-sha-pinned",
                        rel,
                        f"Action {action}@{reference} is not pinned to a full commit SHA",
                    )
                )
        if re.search(r"(?m)^\s{2}pull_request:\s*$", text) and not re.search(
            r"(?m)^\s{2}merge_group:\s*$", text
        ):
            violations.append(
                PolicyViolation(
                    "merge-group-missing",
                    rel,
                    "Pull-request workflow does not handle merge_group",
                )
            )
        jobs_match = re.search(r"(?m)^jobs:\s*$", text)
        if jobs_match:
            jobs_text = text[jobs_match.end() :]
            starts = list(re.finditer(r"(?m)^  ([A-Za-z0-9_-]+):\s*$", jobs_text))
            for index, start in enumerate(starts):
                end = (
                    starts[index + 1].start()
                    if index + 1 < len(starts)
                    else len(jobs_text)
                )
                block = jobs_text[start.end() : end]
                if not re.search(r"(?m)^    timeout-minutes:\s*\d+\s*$", block):
                    violations.append(
                        PolicyViolation(
                            "job-timeout-missing",
                            rel,
                            f"Job {start.group(1)} has no explicit timeout-minutes",
                        )
                    )
    return violations


def _control_violations(root: Path) -> list[PolicyViolation]:
    path = root / ".github" / "assurance-controls.json"
    rel = ".github/assurance-controls.json"
    if not path.is_file():
        return [
            PolicyViolation(
                "assurance-controls-missing",
                rel,
                "Assurance control manifest is missing",
            )
        ]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [PolicyViolation("assurance-controls-invalid", rel, str(exc))]
    violations: list[PolicyViolation] = []
    if manifest.get("schemaVersion") != 1:
        violations.append(
            PolicyViolation(
                "assurance-controls-schema-version",
                rel,
                "Assurance controls must declare schemaVersion 1",
            )
        )
    lifecycle = manifest.get("lifecycleStates", [])
    expected_states = {
        "configured",
        "enabled",
        "executed",
        "passing",
        "deferred",
        "blocked",
    }
    if set(lifecycle) != expected_states:
        violations.append(
            PolicyViolation(
                "assurance-lifecycle-incomplete",
                rel,
                "Assurance lifecycle states are incomplete",
            )
        )
    controls = manifest.get("controls")
    if not isinstance(controls, list) or not controls:
        violations.append(
            PolicyViolation(
                "assurance-controls-empty",
                rel,
                "Assurance controls must contain at least one declaration",
            )
        )
        return violations
    for index, control in enumerate(controls):
        if not isinstance(control, dict):
            missing = REQUIRED_CONTROL_FIELDS
        else:
            missing = REQUIRED_CONTROL_FIELDS - control.keys()
        if missing:
            violations.append(
                PolicyViolation(
                    "assurance-control-incomplete",
                    rel,
                    f"Control {index} is missing: {', '.join(sorted(missing))}",
                )
            )
    return violations


def _codecov_violations(root: Path) -> list[PolicyViolation]:
    violations: list[PolicyViolation] = []
    config = root / "codecov.yml"
    if not config.is_file():
        violations.append(
            PolicyViolation(
                "codecov-config-missing",
                "codecov.yml",
                "Root Codecov configuration is missing",
            )
        )
    workflow_dir = root / ".github" / "workflows"
    codecov_workflows: list[tuple[Path, str]] = []
    for path in sorted((*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml"))):
        text = path.read_text(encoding="utf-8")
        if "codecov/codecov-action@" in text:
            codecov_workflows.append((path, text))
    if not codecov_workflows:
        violations.append(
            PolicyViolation(
                "codecov-upload-missing",
                ".github/workflows",
                "No Codecov upload step is configured",
            )
        )
        return violations
    for path, text in codecov_workflows:
        rel = _relative(path, root)
        if "fail_ci_if_error: true" not in text:
            violations.append(
                PolicyViolation(
                    "codecov-not-blocking",
                    rel,
                    "Codecov upload failures do not block CI",
                )
            )
        if "use_oidc: true" not in text or not re.search(
            r"(?m)^\s+id-token:\s*write\s*$", text
        ):
            violations.append(
                PolicyViolation(
                    "codecov-oidc-missing",
                    rel,
                    "Codecov upload is not authenticated with OIDC",
                )
            )
        if not re.search(
            r"--cov-report=(?:xml(?::coverage\.xml)?|xml:coverage\.xml)", text
        ):
            violations.append(
                PolicyViolation(
                    "coverage-xml-not-generated",
                    rel,
                    "Workflow does not explicitly generate coverage.xml",
                )
            )
    return violations


def _renovate_violations(root: Path) -> list[PolicyViolation]:
    violations: list[PolicyViolation] = []
    path = root / ".github" / "renovate.json"
    if not path.is_file():
        return [
            PolicyViolation(
                "renovate-config-missing",
                ".github/renovate.json",
                "Renovate config is missing",
            )
        ]
    config = json.loads(path.read_text(encoding="utf-8"))
    managers = set(config.get("enabledManagers", []))
    missing = sorted(REQUIRED_RENOVATE_MANAGERS - managers)
    if missing:
        violations.append(
            PolicyViolation(
                "renovate-manager-coverage",
                ".github/renovate.json",
                f"Renovate enabledManagers is missing: {', '.join(missing)}",
            )
        )
    for package_path in sorted(root.rglob("package.json")):
        if any(
            part in {"node_modules", ".venv", ".git"} for part in package_path.parts
        ):
            continue
        package = json.loads(package_path.read_text(encoding="utf-8"))
        for section in (
            "dependencies",
            "devDependencies",
            "optionalDependencies",
            "peerDependencies",
        ):
            for name, specifier in package.get(section, {}).items():
                if isinstance(specifier, str) and LOCAL_SPECIFIER.match(specifier):
                    violations.append(
                        PolicyViolation(
                            "nonportable-local-dependency",
                            _relative(package_path, root),
                            f"Dependency {name} uses non-portable specifier {specifier}",
                        )
                    )
    return violations


def validate_repository(root: Path) -> list[PolicyViolation]:
    """Return all deterministic policy violations below *root*."""
    resolved = root.resolve()
    violations: list[PolicyViolation] = []
    violations.extend(_control_violations(resolved))
    violations.extend(_workflow_violations(resolved))
    violations.extend(_codecov_violations(resolved))
    violations.extend(_renovate_violations(resolved))
    return sorted(set(violations))


def main() -> int:
    """Run policy validation and return non-zero when violations exist."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    violations = validate_repository(Path(args.root))
    if args.as_json:
        print(json.dumps([asdict(item) for item in violations], indent=2))
    else:
        for item in violations:
            print(f"{item.path}: {item.code}: {item.message}")
        print(f"CI policy violations: {len(violations)}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
