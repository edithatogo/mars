#!/bin/bash
cd /Users/doughnut/GitHub/pymars
LOG="/Users/doughnut/GitHub/pymars/.conda.log"

echo "=== STASH ===" > "$LOG"
git stash >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== PULL ===" >> "$LOG"
git pull --rebase origin main >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== POP STASH ===" >> "$LOG"
git stash pop >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== ADD & COMMIT ===" >> "$LOG"
git add -A >> "$LOG" 2>&1
git commit -m "chore: add track for strict ruff, strict typing, and >90% coverage" >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== PUSH ===" >> "$LOG"
git push origin main >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== BASELINE ===" >> "$LOG"
echo "--- Ruff check ---" >> "$LOG"
ruff check pymars/ 2>&1 | tail -5 >> "$LOG" 2>&1
echo "" >> "$LOG"
echo "--- Mypy strict ---" >> "$LOG"
mypy pymars/ --strict 2>&1 | tail -5 >> "$LOG" 2>&1
echo "" >> "$LOG"
echo "--- Coverage ---" >> "$LOG"
python -m pytest --cov=pymars --cov-report=term-missing -q 2>&1 | tail -10 >> "$LOG" 2>&1
echo "" >> "$LOG"

echo "=== DONE ===" >> "$LOG"
