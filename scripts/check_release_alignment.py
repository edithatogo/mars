#!/usr/bin/env python3
"""Check that release docs, package manifests, and canonical release metadata align."""

from __future__ import annotations

import json
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    errors: list[str] = []

    errors.extend(check_canonical_release_metadata())
    errors.extend(check_manifest_names())
    errors.extend(check_release_inventory())
    errors.extend(check_package_paths())

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("release alignment check passed")
    return 0


def check_canonical_release_metadata() -> list[str]:
    errors: list[str] = []
    path = ROOT / "docs/release_metadata.json"
    if not path.exists():
        return ["docs/release_metadata.json is missing"]

    data = json.loads(path.read_text())
    if data.get("brand") != "mars-earth":
        errors.append("docs/release_metadata.json brand must be mars-earth")

    expected = {
        "Python": ("mars-earth", "1.0.4"),
        "Rust": ("mars-earth", "0.1.0"),
        "R": ("marsruntime", "0.0.0"),
        "Julia": ("MarsRuntime", "0.1.0"),
        "C#": ("mars-earth", "0.0.0"),
        "Go": ("github.com/edithatogo/mars/bindings/go", "0.1.0"),
        "TypeScript": ("mars-earth", "0.0.0"),
    }
    packages = {item.get("language"): item for item in data.get("packages", [])}
    for language, (package, version) in expected.items():
        item = packages.get(language)
        if item is None:
            errors.append(f"docs/release_metadata.json missing package entry for {language}")
            continue
        if item.get("package") != package:
            errors.append(f"docs/release_metadata.json {language} package must be {package}")
        if item.get("version") != version:
            errors.append(f"docs/release_metadata.json {language} version must be {version}")

    return errors


def check_manifest_names() -> list[str]:
    errors: list[str] = []

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    if pyproject.get("project", {}).get("name") != "mars-earth":
        errors.append("pyproject.toml project.name must be mars-earth")

    rust = tomllib.loads((ROOT / "rust-runtime/Cargo.toml").read_text())
    if rust.get("package", {}).get("name") != "mars-earth":
        errors.append("rust-runtime/Cargo.toml package.name must be mars-earth")

    typescript = json.loads((ROOT / "bindings/typescript/package.json").read_text())
    if typescript.get("name") != "mars-earth":
        errors.append("bindings/typescript/package.json name must be mars-earth")

    csproj = ET.fromstring((ROOT / "bindings/csharp/MarsRuntime.csproj").read_text())
    package_id = csproj.findtext(".//PackageId")
    if package_id != "mars-earth":
        errors.append("bindings/csharp/MarsRuntime.csproj PackageId must be mars-earth")

    return errors


def check_release_inventory() -> list[str]:
    text = (ROOT / "docs/release_inventory.md").read_text()
    expected = [
        "| Python | `mars-earth` / `pymars` import name |",
        "| Rust | `mars-earth` |",
        "| C# | `mars-earth` |",
        "| TypeScript | `mars-earth` |",
        "| R | `marsruntime` |",
        "| Julia | `MarsRuntime` |",
        "| crates.io `mars-earth` |",
        "| npm `mars-earth` |",
        "| NuGet `mars-earth` |",
    ]
    return [f"docs/release_inventory.md missing expected row: {row}" for row in expected if row not in text]


def check_package_paths() -> list[str]:
    text = (ROOT / "docs/package_release_paths.md").read_text()
    expected = [
        "Package name: `mars-earth`.",
        "Package name: `marsruntime`.",
        "Package name: `MarsRuntime`.",
    ]

    errors = [f"docs/package_release_paths.md missing expected line: {line}" for line in expected if line not in text]

    sections = {
        "Rust": "Package name: `mars-earth`.",
        "TypeScript": "Package name: `mars-earth`.",
        "C#": "Package name: `mars-earth`.",
    }
    for section, expected_line in sections.items():
        if section not in text or expected_line not in text:
            errors.append(f"docs/package_release_paths.md missing {section} release path details")

    return errors


if __name__ == "__main__":
    raise SystemExit(main())
