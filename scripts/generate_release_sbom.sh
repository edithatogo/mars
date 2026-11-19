#!/usr/bin/env bash
# Retain the exact distributions and the installed, frozen Python dependency graph.
set -euo pipefail
shopt -s nullglob
wheels=(dist/*.whl)
sdists=(dist/*.tar.gz)
if [ "${#wheels[@]}" -ne 1 ] || [ "${#sdists[@]}" -ne 1 ]; then
  echo 'Expected exactly one wheel and one source distribution.' >&2
  exit 1
fi
test ! -e .grype-runtime
mkdir -p release-evidence .grype-rust-inputs
sha256sum "${wheels[@]}" "${sdists[@]}" > release-evidence/subjects.sha256
git rev-parse HEAD > release-evidence/source-commit.txt
uv export --frozen --no-dev --no-emit-project --extra cli --extra pandas \
  --format requirements-txt --output-file release-evidence/runtime-requirements.txt
uv venv --python 3.12 .grype-runtime
uv pip install --python .grype-runtime/bin/python --require-hashes \
  -r release-evidence/runtime-requirements.txt
uv pip install --python .grype-runtime/bin/python --no-deps "${wheels[0]}"
uv pip check --python .grype-runtime/bin/python
uvx --from cyclonedx-bom==7.3.0 cyclonedx-py environment \
  .grype-runtime/bin/python --pyproject pyproject.toml --mc-type library \
  --sv 1.6 --output-reproducible --of JSON --validate \
  -o release-evidence/python.sbom.cdx.json
# Keep the resolved Cargo inventory separate from Python's installed packages.
cp rust-runtime/Cargo.lock .grype-rust-inputs/Cargo.lock
sha256sum rust-runtime/Cargo.lock release-evidence/python.sbom.cdx.json \
  >> release-evidence/subjects.sha256
sha256sum --check release-evidence/subjects.sha256
