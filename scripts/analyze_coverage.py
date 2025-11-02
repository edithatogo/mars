#!/usr/bin/env python
"""
Script to analyze test coverage and identify modules that need better coverage.
"""
import subprocess
import sys


def analyze_coverage():
    """Run coverage analysis and identify low-coverage modules."""
    try:
        # Run coverage command to get detailed report
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/",
            "--cov=pymars", "--cov-report=term-missing", "-q"
        ], capture_output=True, text=True, cwd=".")

        output = result.stdout + result.stderr

        # Parse the coverage report
        lines = output.split('\n')
        coverage_started = False
        coverage_data = []

        for line in lines:
            if 'Name' in line and 'Stmts' in line and 'Miss' in line and 'Cover' in line:
                coverage_started = True
                continue
            if coverage_started and line.strip() and not line.startswith('-') and 'TOTAL' not in line:
                coverage_data.append(line.strip())
            elif 'TOTAL' in line:
                break

        print("Coverage Analysis Report")
        print("=" * 80)
        print(f"{'Module':<35} {'Stmts':<6} {'Miss':<6} {'Cover':<6} Missing")
        print("-" * 80)

        low_coverage_modules = []

        for line in coverage_data:
            parts = line.split()
            if len(parts) >= 4:
                module_path = parts[0]
                stmts = parts[1] if len(parts) > 1 else "0"
                miss = parts[2] if len(parts) > 2 else "0"
                cover = parts[3] if len(parts) > 3 else "0%"

                # Extract the missing lines if available
                missing = " ".join(parts[4:]) if len(parts) > 4 else ""

                print(f"{module_path:<35} {stmts:<6} {miss:<6} {cover:<6} {missing}")

                # Check if coverage is below 90%
                try:
                    cover_pct = int(cover.rstrip('%'))
                    if cover_pct < 90:
                        low_coverage_modules.append((module_path, cover_pct, missing))
                except ValueError:
                    pass

        print("\nModules with coverage < 90%:")
        print("-" * 50)
        for module, pct, missing in low_coverage_modules:
            print(f"  {module}: {pct}% - Missing: {missing}")

        print(f"\nTotal low-coverage modules: {len(low_coverage_modules)}")

        if low_coverage_modules:
            print("\nRecommendation: Write additional tests for the above modules to achieve >90% coverage.")
            return False
        else:
            print("\nAll modules have >=90% coverage!")
            return True

    except Exception as e:
        print(f"Error running coverage analysis: {e}")
        return False

if __name__ == "__main__":
    success = analyze_coverage()
    sys.exit(0 if success else 1)
