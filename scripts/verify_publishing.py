#!/usr/bin/env python
"""
Test publishing script for pymars.

This script verifies that the package is ready for publishing to TestPyPI.
It checks that all necessary files are present and correctly formatted.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_distribution_files():
    """Check that distribution files are present and valid."""
    print("=" * 60)
    print("📦 CHECKING DISTRIBUTION FILES")
    print("=" * 60)
    
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("❌ Distribution directory not found")
        return False
    
    dist_files = list(dist_dir.glob("*"))
    if not dist_files:
        print("❌ No distribution files found")
        return False
    
    # Check for wheel and source distribution
    wheel_files = list(dist_dir.glob("*.whl"))
    source_files = list(dist_dir.glob("*.tar.gz"))
    
    if not wheel_files:
        print("❌ No wheel files found")
        return False
    
    if not source_files:
        print("❌ No source distribution files found")
        return False
    
    print(f"✅ Wheel files found: {len(wheel_files)}")
    for wf in wheel_files:
        print(f"   - {wf.name} ({wf.stat().st_size / 1024:.1f}KB)")
    
    print(f"✅ Source distribution files found: {len(source_files)}")
    for sf in source_files:
        print(f"   - {sf.name} ({sf.stat().st_size / 1024:.1f}KB)")
    
    return True


def check_metadata():
    """Check that package metadata is correct."""
    print("\\n" + "=" * 60)
    print("📋 CHECKING PACKAGE METADATA")
    print("=" * 60)
    
    # Check pyproject.toml
    pyproject_file = Path("pyproject.toml")
    if not pyproject_file.exists():
        print("❌ pyproject.toml not found")
        return False
    
    # Read and verify version
    with open(pyproject_file, "r") as f:
        content = f.read()
        if "version = \"1.0.0\"" not in content:
            print("❌ Version not set to 1.0.0 in pyproject.toml")
            return False
    
    print("✅ pyproject.toml found and version is 1.0.0")
    
    # Check __init__.py
    init_file = Path("pymars/__init__.py")
    if not init_file.exists():
        print("❌ pymars/__init__.py not found")
        return False
    
    # Read and verify version
    with open(init_file, "r") as f:
        content = f.read()
        if "__version__ = \"1.0.0\"" not in content:
            print("❌ Version not set to 1.0.0 in pymars/__init__.py")
            return False
    
    print("✅ pymars/__init__.py found and version is 1.0.0")
    
    return True


def check_readme():
    """Check that README is present and properly formatted."""
    print("\\n" + "=" * 60)
    print("📄 CHECKING README")
    print("=" * 60)
    
    readme_file = Path("README.md")
    if not readme_file.exists():
        print("❌ README.md not found")
        return False
    
    # Check that README has content
    with open(readme_file, "r") as f:
        content = f.read()
        if len(content) < 100:
            print("❌ README.md appears to be too short")
            return False
    
    print("✅ README.md found and has sufficient content")
    return True


def check_license():
    """Check that LICENSE is present."""
    print("\\n" + "=" * 60)
    print("⚖️ CHECKING LICENSE")
    print("=" * 60)
    
    license_file = Path("LICENSE")
    if not license_file.exists():
        print("❌ LICENSE file not found")
        return False
    
    # Check that LICENSE has content
    with open(license_file, "r") as f:
        content = f.read()
        if len(content) < 50:
            print("❌ LICENSE appears to be too short")
            return False
    
    print("✅ LICENSE file found and has content")
    return True


def check_requirements():
    """Check that requirements are properly specified."""
    print("\\n" + "=" * 60)
    print("🧾 CHECKING REQUIREMENTS")
    print("=" * 60)
    
    # Check pyproject.toml dependencies
    pyproject_file = Path("pyproject.toml")
    with open(pyproject_file, "r") as f:
        content = f.read()
        
        required_deps = ["numpy", "scikit-learn", "matplotlib"]
        for dep in required_deps:
            if dep not in content:
                print(f"❌ Required dependency '{dep}' not found in pyproject.toml")
                return False
    
    print("✅ All required dependencies found in pyproject.toml")
    return True


def check_build_system():
    """Check that build system is properly configured."""
    print("\\n" + "=" * 60)
    print("🏗️ CHECKING BUILD SYSTEM")
    print("=" * 60)
    
    # Try to import build tools
    try:
        import build
        print("✅ build module available")
    except ImportError:
        print("❌ build module not available")
        return False
    
    try:
        import twine
        print("✅ twine module available")
    except ImportError:
        print("❌ twine module not available")
        return False
    
    return True


def simulate_publish_to_testpypi():
    """Simulate the publishing process to TestPyPI."""
    print("\\n" + "=" * 60)
    print("🧪 SIMULATING PUBLISH TO TESTPYPI")
    print("=" * 60)
    
    # Check if we have the required files
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("❌ Distribution directory not found")
        return False
    
    dist_files = list(dist_dir.glob("*"))
    if not dist_files:
        print("❌ No distribution files found")
        return False
    
    print("✅ Distribution files ready for upload:")
    for df in dist_files:
        print(f"   - {df.name}")
    
    # Check if twine is available
    try:
        result = subprocess.run(["twine", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Twine is available for publishing")
        else:
            print("❌ Twine is not available")
            return False
    except FileNotFoundError:
        print("❌ Twine is not installed")
        return False
    
    # Show what the publish command would be
    print("\\n📝 Publish command (dry run):")
    print("   twine upload --repository testpypi dist/*")
    print("\\n📝 To publish to TestPyPI, you would need:")
    print("   1. A TestPyPI account: https://test.pypi.org/account/register/")
    print("   2. An API token from your TestPyPI account")
    print("   3. A .pypirc file with your credentials")
    print("   4. To run: twine upload --repository testpypi dist/*")
    
    return True


def main():
    """Run all checks."""
    print("🚀 pymars v1.0.0 Publishing Verification")
    print("=" * 60)
    
    checks = [
        check_distribution_files,
        check_metadata,
        check_readme,
        check_license,
        check_requirements,
        check_build_system,
        simulate_publish_to_testpypi
    ]
    
    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
    
    print("\\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL PUBLISHING CHECKS PASSED!")
        print("pymars v1.0.0 is READY FOR PUBLISHING TO TESTPYPI!")
        print("=" * 60)
        print("\\n📋 Next Steps:")
        print("1. Create a .pypirc file with your TestPyPI credentials")
        print("2. Run: twine upload --repository testpypi dist/*")
        print("3. Test installation: pip install --index-url https://test.pypi.org/simple/ pymars")
        return True
    else:
        print("❌ SOME PUBLISHING CHECKS FAILED!")
        print("pymars v1.0.0 needs fixes before publishing to TestPyPI.")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)