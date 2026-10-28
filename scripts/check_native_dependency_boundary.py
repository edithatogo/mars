"""Reject Python ABI dependencies in the normal and build graphs of a native core."""
import argparse
import json
from pathlib import Path
import subprocess


def check(manifest: Path) -> list[str]:
    result = subprocess.run(
        ['cargo', 'tree', '--manifest-path', str(manifest), '--locked', '--offline',
         '-e', 'normal,build', '--prefix', 'none'],
        check=True, capture_output=True, text=True,
    )
    return [line for line in result.stdout.splitlines()
            if line.startswith(('pyo3 ', 'pyo3-'))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'rust-runtime/Cargo.toml')
    args = parser.parse_args()
    dependencies = check(args.manifest)
    print(json.dumps({'default_core_python_dependencies': dependencies}))
    return int(bool(dependencies))


if __name__ == '__main__':
    raise SystemExit(main())
