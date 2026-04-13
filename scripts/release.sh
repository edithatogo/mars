#!/bin/bash
cd /Users/doughnut/GitHub/pymars
LOG="/Users/doughnut/GitHub/pymars/.release.log"

echo "=== GIT STATUS ===" > "$LOG"
git status >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== GIT LOG -3 ===" >> "$LOG"
git log -3 --oneline >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== ADD ===" >> "$LOG"
git add -A >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== COMMIT ===" >> "$LOG"
git commit -m "feat: rename package to mars-earth, add TestPyPI gate, conda-forge recipe, and post-publish verification" >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== PUSH ===" >> "$LOG"
git push origin main >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== TAG ===" >> "$LOG"
git tag -a v1.0.1 -m "Release v1.0.1 - mars-earth package rename with full publishing pipeline" >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== PUSH TAG ===" >> "$LOG"
git push origin v1.0.1 >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== DONE ===" >> "$LOG"
