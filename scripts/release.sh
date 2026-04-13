#!/bin/bash
cd /Users/doughnut/GitHub/pymars
LOG="/Users/doughnut/GitHub/pymars/.release.log"

echo "=== STATUS ===" > "$LOG"
git status >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== STASH ===" >> "$LOG"
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
git commit -m "fix(release): add --system flag to uv pip install in smoke-test" >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== PUSH ===" >> "$LOG"
git push origin main >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== DELETE TAG ===" >> "$LOG"
git tag -d v1.0.1 >> "$LOG" 2>&1
git push --delete origin v1.0.1 >> "$LOG" 2>&1
echo "" >> "$LOG"

sleep 5

echo "=== CREATE TAG ===" >> "$LOG"
git tag -a v1.0.1 -m "Release v1.0.1 - mars-earth package with full publishing pipeline" >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== PUSH TAG ===" >> "$LOG"
git push origin v1.0.1 >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== DONE ===" >> "$LOG"
