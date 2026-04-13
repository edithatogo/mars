#!/bin/bash
cd /Users/doughnut/GitHub/pymars
LOG="/Users/doughnut/GitHub/pymars/.release.log"

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
git commit -m "chore(release): bump to v1.0.2 (v1.0.1 already on PyPI)" >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== PUSH ===" >> "$LOG"
git push origin main >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== DELETE OLD TAG ===" >> "$LOG"
git tag -d v1.0.1 >> "$LOG" 2>&1
git push --delete origin v1.0.1 >> "$LOG" 2>&1
echo "" >> "$LOG"

sleep 3

echo "=== CREATE v1.0.2 TAG ===" >> "$LOG"
git tag -a v1.0.2 -m "Release v1.0.2 - mars-earth on PyPI and TestPyPI" >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== PUSH TAG ===" >> "$LOG"
git push origin v1.0.2 >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== DONE ===" >> "$LOG"
