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


@dataclass(frozen=True, order=True)
class PolicyViolation:
    """One deterministic CI policy violation."""

    code: str
    path: str
    message: str


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _workflow_violations(root: Path) -> list[PolicyViolation]:
    violations: list[PolicyViolation] = []
    workflow_dir = root / ".github" / "workflows"
    workflows = sorted((*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")))
    for path in workflows:
        rel = _relative(path, root)
        text = path.read_text(encoding="utf-8")
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


def _go_version_violations(root: Path) -> list[PolicyViolation]:
    module = root / "go.mod"
    if not module.is_file():
        return []
    match = re.search(
        r"(?m)^go\s+([0-9]+(?:\.[0-9]+)+)\s*$", module.read_text(encoding="utf-8")
    )
    if match is None:
        return []
    governed = match.group(1)
    violations: list[PolicyViolation] = []
    workflow_dir = root / ".github" / "workflows"
    for path in sorted((*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml"))):
        text = path.read_text(encoding="utf-8")
        declared = {
            *re.findall(r"go-version:\s*[\"']?([0-9]+(?:\.[0-9]+)+)", text),
            *re.findall(r"(?m)^\s+go\s+([0-9]+(?:\.[0-9]+)+)\s*$", text),
        }
        stale = sorted(version for version in declared if version != governed)
        if stale:
            violations.append(
                PolicyViolation(
                    "go-version-drift",
                    _relative(path, root),
                    f"Go versions {', '.join(stale)} do not match go.mod floor {governed}",
                )
            )
    return violations


def validate_repository(root: Path) -> list[PolicyViolation]:
    """Return all deterministic policy violations below *root*."""
    resolved = root.resolve()
    controls = resolved / ".github" / "assurance-controls.json"
    violations: list[PolicyViolation] = []
    if not controls.is_file():
        violations.append(
            PolicyViolation(
                "assurance-controls-missing",
                ".github/assurance-controls.json",
                "Assurance control manifest is missing",
            )
        )
    violations.extend(_workflow_violations(resolved))
    violations.extend(_codecov_violations(resolved))
    violations.extend(_renovate_violations(resolved))
    violations.extend(_go_version_violations(resolved))
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
