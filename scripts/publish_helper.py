#!/usr/bin/env python
"""
Helper script for publishing pymars to TestPyPI and PyPI.

This script automates the publishing process and provides helpful guidance.
"""

import os
import sys
import subprocess
import getpass
from pathlib import Path


def check_prerequisites():
    """Check if all prerequisites are met for publishing."""
    print("🔍 Checking publishing prerequisites...")
    
    # Check if build tools are available
    try:
        subprocess.run([sys.executable, "-m", "build", "--version"], 
                      capture_output=True, check=True)
        print("✅ Build tools available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Build tools not available. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "build"], check=True)
    
    # Check if twine is available
    try:
        subprocess.run(["twine", "--version"], capture_output=True, check=True)
        print("✅ Twine available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Twine not available. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "twine"], check=True)
    
    # Check if dist directory exists and has files
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("❌ Dist directory not found")
        return False
    
    dist_files = list(dist_dir.glob("*"))
    if not dist_files:
        print("❌ No distribution files found")
        return False
    
    print(f"✅ Found {len(dist_files)} distribution files")
    for f in dist_files:
        print(f"   - {f.name}")
    
    return True


def setup_pypirc():
    """Setup .pypirc file for authentication."""
    print("\\n🔐 Setting up .pypirc authentication...")
    
    home_dir = Path.home()
    pypirc_path = home_dir / ".pypirc"
    
    if pypirc_path.exists():
        print("✅ .pypirc file already exists")
        return True
    
    print("📝 Creating .pypirc file...")
    print("   You'll need your PyPI/TestPyPI API tokens for this.")
    print("   If you don't have tokens, register at:")
    print("   - PyPI: https://pypi.org/account/register/")
    print("   - TestPyPI: https://test.pypi.org/account/register/")
    
    # Ask for tokens
    print("\\nPlease enter your API tokens:")
    test_token = getpass.getpass("TestPyPI token (leave blank if not available): ")
    prod_token = getpass.getpass("PyPI token (leave blank if not available): ")
    
    if not test_token and not prod_token:
        print("❌ No tokens provided. Cannot create .pypirc file.")
        return False
    
    # Create .pypirc content
    pypirc_content = "[distutils]\nindex-servers =\n"
    if test_token:
        pypirc_content += "    testpypi\n"
    if prod_token:
        pypirc_content += "    pypi\n"
    pypirc_content += "\n"
    
    if prod_token:
        pypirc_content += "[pypi]\n"
        pypirc_content += "username = __token__\n"
        pypirc_content += f"password = {prod_token}\n\n"
    
    if test_token:
        pypirc_content += "[testpypi]\n"
        pypirc_content += "repository = https://test.pypi.org/legacy/\n"
        pypirc_content += "username = __token__\n"
        pypirc_content += f"password = {test_token}\n\n"
    
    # Write .pypirc file
    try:
        with open(pypirc_path, "w") as f:
            f.write(pypirc_content)
        os.chmod(pypirc_path, 0o600)  # Set secure permissions
        print("✅ .pypirc file created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create .pypirc file: {e}")
        return False


def publish_to_testpypi():
    """Publish to TestPyPI."""
    print("\\n🚀 Publishing to TestPyPI...")
    
    try:
        result = subprocess.run([
            "twine", "upload", 
            "--repository", "testpypi",
            "dist/*"
        ], check=True)
        print("✅ Successfully published to TestPyPI!")
        print("   Test installation with:")
        print("   pip install --index-url https://test.pypi.org/simple/ pymars")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to publish to TestPyPI: {e}")
        return False


def publish_to_pypi():
    """Publish to PyPI."""
    print("\\n🚀 Publishing to PyPI...")
    
    try:
        result = subprocess.run([
            "twine", "upload", 
            "dist/*"
        ], check=True)
        print("✅ Successfully published to PyPI!")
        print("   Installation with:")
        print("   pip install pymars")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to publish to PyPI: {e}")
        return False


def test_installation():
    """Test the installation of the package."""
    print("\\n🧪 Testing package installation...")
    
    # Test installation in a temporary environment
    try:
        # Create a temporary test
        test_script = '''
import pymars as pm
import numpy as np

print("🧪 Testing pymars installation...")
np.random.seed(42)
X = np.random.rand(10, 2)
y = X[:, 0] + X[:, 1] * 0.5

model = pm.Earth(max_degree=2, penalty=3.0, max_terms=10)
model.fit(X, y)
score = model.score(X, y)
print(f"✅ Earth model R²: {score:.4f}")

regressor = pm.EarthRegressor(max_degree=2, penalty=3.0, max_terms=10)
regressor.fit(X, y)
reg_score = regressor.score(X, y)
print(f"✅ EarthRegressor R²: {reg_score:.4f}")

print("🎉 Installation test passed!")
'''
        
        result = subprocess.run([
            sys.executable, "-c", test_script
        ], capture_output=True, text=True, check=True)
        
        print(result.stdout.strip())
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation test failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    """Main function to run the publishing helper."""
    print("=" * 60)
    print("🚀 pymars Publishing Helper")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\\n❌ Prerequisites not met. Please fix issues and try again.")
        return False
    
    # Setup authentication
    print("\\n" + "=" * 60)
    if not setup_pypirc():
        print("\\n❌ Authentication setup failed.")
        return False
    
    # Ask user what they want to do
    print("\\n" + "=" * 60)
    print("📋 What would you like to do?")
    print("1. Publish to TestPyPI (recommended for testing first)")
    print("2. Publish to PyPI (production release)")
    print("3. Test installation")
    print("4. Exit")
    
    choice = input("\\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        return publish_to_testpypi()
    elif choice == "2":
        # Confirm before publishing to production
        confirm = input("\\n⚠️  Are you sure you want to publish to PRODUCTION PyPI? (yes/no): ").strip().lower()
        if confirm == "yes":
            return publish_to_pypi()
        else:
            print("❌ Publication cancelled.")
            return False
    elif choice == "3":
        return test_installation()
    elif choice == "4":
        print("👋 Exiting...")
        return True
    else:
        print("❌ Invalid choice.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\\n🎉 Publishing helper completed successfully!")
        else:
            print("\\n❌ Publishing helper encountered issues.")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\\n\\n👋 Publishing helper interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        sys.exit(1)