#!/bin/bash
set -euo pipefail

cd /Users/doughnut/GitHub/pymars
LOG="/Users/doughnut/GitHub/pymars/.release.log"

if [ $# -gt 1 ]; then
  echo "Usage: $0 [version]" >&2
  exit 1
fi

if [ $# -eq 1 ]; then
  VERSION="$1"
else
  VERSION="$(python3 - <<'PY'
from pathlib import Path
import re

match = re.search(r'^version\s*=\s*"([^"]+)"$', Path("pyproject.toml").read_text(), re.M)
if not match:
    raise SystemExit("could not determine version from pyproject.toml")
print(match.group(1))
PY
)"
fi

TAG="v${VERSION}"

echo "=== STASH ===" > "$LOG"
git stash >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== PULL ===" >> "$LOG"
git pull --rebase origin main >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== POP STASH ===" >> "$LOG"
git stash pop >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== ADD ALL ===" >> "$LOG"
git add -A >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== COMMIT ===" >> "$LOG"
git commit -m "chore(release): bump to ${TAG}, add skip-existing to PyPI publish" >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== PUSH ===" >> "$LOG"
git push origin main >> "$LOG" 2>&1
echo "" >> "$LOG"

sleep 3

echo "=== CREATE ${TAG} TAG ===" >> "$LOG"
git tag -a "${TAG}" -m "Release ${TAG} - mars-earth on PyPI" >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== PUSH TAG ===" >> "$LOG"
git push origin "${TAG}" >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== DONE ===" >> "$LOG"
