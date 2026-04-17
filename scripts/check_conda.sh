#!/bin/bash
cd /Users/doughnut/GitHub/pymars
uv run pytest --cov=pymars --cov-report=term 2>&1 | grep -E "TOTAL|passed|FAILED"
