#!/bin/bash
# Script to check test coverage and ensure >90% for each file

echo "Running tests with coverage..."
python -m pytest tests/ --cov=pymars --cov-report=term-missing --cov-report=html

echo "Checking coverage report..."
python -c "
import coverage
import sys

cov = coverage.Coverage()
cov.load()

# Get coverage data
analysis = cov.analysis()
results = cov.report()

# Check if overall coverage is > 90%
if results < 90.0:
    print(f'ERROR: Overall coverage is {results}%, which is less than 90%')
    sys.exit(1)

print(f'Overall coverage: {results}% - PASSED')

# Analyze by file
print('\\nFile-by-file coverage:')
for filename, statements, excluded, missing, _ in cov.get_data().measured_files():
    stmts = set(statements)
    miss = set(missing)
    covered_count = len(stmts - miss)
    total_count = len(stmts)
    
    if total_count > 0:
        file_cov = (covered_count / total_count) * 100
        status = 'PASSED' if file_cov >= 90 else 'FAILED'
        print(f'{filename}: {file_cov:.1f}% {status}')
        
        if file_cov < 90:
            print(f'  Missing lines: {sorted(missing)}')
            sys.exit(1)
    else:
        print(f'{filename}: No executable statements')

print('\\nAll files have >90% coverage - PASSED')
"