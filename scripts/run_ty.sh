#!/bin/bash
cd /Users/doughnut/GitHub/pymars
LOG="/Users/doughnut/GitHub/pymars/.ty_full4.log"
uv run ty check pymars/ > "$LOG" 2>&1
echo "=== Summary ===" >> "$LOG"
echo "Total diagnostics:" >> "$LOG"
grep -c "^error\[" "$LOG" >> "$LOG" 2>/dev/null || echo "0" >> "$LOG"
echo "" >> "$LOG"
echo "By file:" >> "$LOG"
grep "^  -->" "$LOG" | sed 's/.*--> //;s/:.*//' | sort | uniq -c | sort -rn >> "$LOG" 2>/dev/null
echo "DONE"
