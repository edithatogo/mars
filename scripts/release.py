#!/usr/bin/env python
"""
Release script for pymars.

This script automates the release process for pymars, including:
1. Version bumping
2. Building the package
3. Creating git tags
4. Publishing to PyPI/TestPyPI
5. Creating GitHub releases

Usage:
    python scripts/release.py [--test] [--version VERSION]
    
Options:
    --test              Publish to TestPyPI instead of PyPI
    --version VERSION   Specify the version to release (otherwise uses current)
"""

import argparse
import subprocess
import sys
import os


def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        sys.exit(result.returncode)
    return result


def get_current_version():
    """Get the current version from __init__.py."""
    init_file = os.path.join(os.path.dirname(__file__), "..", "pymars", "__init__.py")
    with open(init_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"')
    return None


def bump_version(version, part="patch"):
    """Bump the version number."""
    parts = version.split(".")
    if part == "major":
        parts[0] = str(int(parts[0]) + 1)
        parts[1] = "0"
        parts[2] = "0"
    elif part == "minor":
        parts[1] = str(int(parts[1]) + 1)
        parts[2] = "0"
    elif part == "patch":
        parts[2] = str(int(parts[2]) + 1)
    return ".".join(parts)


def update_version_file(new_version):
    """Update the version in __init__.py."""
    init_file = os.path.join(os.path.dirname(__file__), "..", "pymars", "__init__.py")
    with open(init_file, "r") as f:
        lines = f.readlines()
    
    with open(init_file, "w") as f:
        for line in lines:
            if line.startswith("__version__"):
                f.write(f'__version__ = "{new_version}"\n')
            else:
                f.write(line)


def update_pyproject_version(new_version):
    """Update the version in pyproject.toml."""
    pyproject_file = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject_file, "r") as f:
        lines = f.readlines()
    
    with open(pyproject_file, "w") as f:
        for line in lines:
            if line.startswith("version ="):
                f.write(f'version = "{new_version}"\n')
            else:
                f.write(line)


def build_package():
    """Build the package."""
    # Clean previous builds
    run_command("rm -rf dist/ build/ *.egg-info", check=False)
    
    # Build the package
    run_command("python -m build")


def create_git_tag(version):
    """Create a git tag for the release."""
    run_command(f"git add .")
    run_command(f'git commit -m "chore: Release v{version}"', check=False)
    run_command(f"git tag -a v{version} -m 'Release v{version}'")
    run_command(f"git push origin main --tags")


def publish_to_pypi(test=False):
    """Publish the package to PyPI or TestPyPI."""
    if test:
        run_command("twine upload --repository testpypi dist/*")
    else:
        run_command("twine upload dist/*")


def create_github_release(version, test=False):
    """Create a GitHub release."""
    if test:
        title = f"pymars v{version}-beta"
        notes = f"Beta release of pymars v{version}"
    else:
        title = f"pymars v{version}"
        notes = f"Stable release of pymars v{version}"
    
    # Get the built files
    dist_files = []
    for file in os.listdir("dist"):
        if file.endswith(".whl") or file.endswith(".tar.gz"):
            dist_files.append(f"dist/{file}")
    
    if dist_files:
        files_arg = " ".join(dist_files)
        run_command(f"gh release create v{version} --title '{title}' --notes '{notes}' {files_arg}")
    else:
        run_command(f"gh release create v{version} --title '{title}' --notes '{notes}'")


def main():
    parser = argparse.ArgumentParser(description="Release script for pymars")
    parser.add_argument("--test", action="store_true", help="Publish to TestPyPI instead of PyPI")
    parser.add_argument("--version", help="Specify the version to release")
    parser.add_argument("--bump", choices=["major", "minor", "patch"], help="Bump version automatically")
    
    args = parser.parse_args()
    
    # Get current version
    current_version = get_current_version()
    print(f"Current version: {current_version}")
    
    # Determine new version
    if args.version:
        new_version = args.version
    elif args.bump:
        new_version = bump_version(current_version, args.bump)
    else:
        new_version = current_version
    
    print(f"Releasing version: {new_version}")
    
    # Update version files if needed
    if new_version != current_version:
        print("Updating version files...")
        update_version_file(new_version)
        update_pyproject_version(new_version)
    
    # Build the package
    print("Building package...")
    build_package()
    
    # Create git tag
    print("Creating git tag...")
    create_git_tag(new_version)
    
    # Publish to PyPI/TestPyPI
    print("Publishing to PyPI...")
    try:
        publish_to_pypi(test=args.test)
    except Exception as e:
        print(f"Failed to publish to PyPI: {e}")
        print("You may need to configure authentication with twine.")
    
    # Create GitHub release
    print("Creating GitHub release...")
    try:
        create_github_release(new_version, test=args.test)
    except Exception as e:
        print(f"Failed to create GitHub release: {e}")
        print("You may need to configure GitHub CLI authentication.")
    
    print("Release process completed!")


if __name__ == "__main__":
    main()