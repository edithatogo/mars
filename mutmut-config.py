# Configuration for mutmut mutation testing
# This file configures mutmut for pymars

# Paths to include for mutation testing
paths_to_mutate = [
    "pymars/*.py",
    "pymars/*/*.py"
]

# Paths to exclude from mutation testing
paths_to_exclude = [
    "tests/",
    "docs/",
    "scripts/",
    "__init__.py",
]

# Files to exclude from mutation testing (glob patterns)
exclude = [
    "*/__init__.py",
    "*/tests/*",
    "*/conftest.py",
]

# Set the runner command to use pytest
runner = "python -m pytest"

# Arguments to pass to the test runner
# Only run tests that are affected by the mutated code
test_command = "python -m pytest tests/ -x --tb=short"

# Configuration for specific mutmut behavior
backup = True  # Whether to create backup files
