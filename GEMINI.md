# Notes for Gemini Models Working on `pymars`

This document contains specific notes, tips, or instructions relevant to Gemini models contributing to the `pymars` project.

## General Approach

*   **Understand the Context:** Before generating code, make sure you understand the specific task, its relation to the overall project (`ROADMAP.md`), and the guidelines in `AGENTS.md`.
*   **Iterative Refinement:** Be prepared to iterate on solutions. Your first attempt might not be perfect. Use feedback and self-correction to improve.
*   **Prioritize Clarity and Correctness:** While `py-earth` (Cython) is performance-oriented, our primary goal for `pymars` (Pure Python) is clarity, correctness, and scikit-learn compatibility. Pythonic performance optimizations can be addressed later if identified as bottlenecks.

## Code Generation

*   **Pure Python:** Remember, all code must be pure Python. Do not generate Cython or C code.
*   **Scikit-learn Compatibility:** Pay close attention to the scikit-learn API requirements outlined in `AGENTS.md`. This includes method signatures, inheritance, and the use of scikit-learn validation utilities.
*   **Type Hinting:** Include type hints in your Python code.
*   **Docstrings:** Generate comprehensive docstrings for all modules, classes, functions, and methods. Google or NumPy style is preferred.
*   **Error Handling:** Implement robust error handling (e.g., `try-except` blocks for expected issues, informative error messages).
*   **`py-earth` Logic Translation:** When translating logic from `py-earth`'s Cython code:
    *   Focus on the algorithmic logic, not a line-by-line translation of Cython-specific syntax.
    *   Python's dynamic typing and high-level features may allow for more concise implementations in some cases.
    *   Be mindful of differences in how NumPy arrays are handled compared to C arrays or typed memoryviews in Cython.
*   **No `goto`:** `py-earth`'s Cython code occasionally uses `goto` for historical reasons or specific optimizations. Avoid this pattern in Python; use loops, conditionals, and functions to structure control flow.

## Tool Usage

*   **`replace_with_git_merge_diff`:**
    *   Be very precise with the `SEARCH` block. Small differences (whitespace, comments) can cause the search to fail.
    *   Ensure the `REPLACE` block correctly integrates with the surrounding code.
*   **`read_files`:**
    *   Use this to understand existing code before making changes.
    *   Request multiple files at once if needed to get a complete picture.
*   **`run_in_bash_session`:**
    *   Use this to run tests (e.g., `pytest tests/`).
    *   Use for linting if a linter is set up.

## Testing

*   When implementing a feature, think about how to test it.
*   If asked to write tests, ensure they cover various scenarios, including edge cases.
*   If tests fail, analyze the output carefully to understand the cause.

## Asking for Clarification

*   If a request is ambiguous or if you foresee significant challenges in meeting conflicting constraints, it's better to ask for clarification (`request_user_input`) than to produce incorrect or suboptimal results.

## Session Logging

*   When making significant progress or complex changes, remember to prompt for an update to `SESSION_LOGS.md`.

By keeping these points in mind, you can contribute effectively to the `pymars` project. Let's build a great library!
