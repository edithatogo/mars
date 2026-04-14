#!/bin/bash
cd /Users/doughnut/GitHub/pymars
LOG="/Users/doughnut/GitHub/pymars/.conda.log"

echo "=== ADD ALL ===" > "$LOG"
git add -A >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== COMMIT ===" >> "$LOG"
git commit -m "chore: add track for strict ruff, strict typing, and >90% coverage" >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== PUSH ===" >> "$LOG"
git push origin main >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== CHECK CONDA WORKFLOW ===" >> "$LOG"
gh run view 24392895876 >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== DONE ===" >> "$LOG"
