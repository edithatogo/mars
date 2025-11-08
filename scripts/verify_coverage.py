#!/usr/bin/env python
"""
Coverage verification script for pymars.

This script verifies that individual file coverage is >90%.
"""
import os
import tempfile
import subprocess
import sys


def verify_coverage():
    """Verify coverage for individual files."""
    print("🔍 Verifying coverage for individual files...")
    print("=" * 60)
    
    # Create a temporary coverage directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run coverage with XML output for detailed per-file analysis
        cmd = [
            sys.executable, "-m", "coverage", "run",
            "--source=pymars/", 
            "-m", "pytest", "tests/", "-v", "--tb=line"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Test execution failed: {result.stderr}")
            return False
        
        # Generate coverage report
        subprocess.run([sys.executable, "-m", "coverage", "report", "--show-missing"], check=True)
        
        # Generate XML coverage report
        subprocess.run([sys.executable, "-m", "coverage", "xml"], check=True)
        
        # Read the coverage XML file to check individual file coverage
        try:
            with open("coverage.xml", "r") as f:
                xml_content = f.read()
                
            # Look for individual file coverage percentages
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_content)
            
            print("\\n📊 Individual File Coverage Analysis:")
            print("-" * 40)
            
            low_coverage_files = []
            total_files = 0
            
            for clazz in root.findall(".//class"):
                filename = clazz.get("filename")
                line_rate = float(clazz.get("line-rate", 0))
                total_files += 1
                
                if line_rate < 0.90:
                    low_coverage_files.append((filename, line_rate))
                
                coverage_pct = line_rate * 100
                status = "✅" if line_rate >= 0.90 else "⚠️ "
                print(f"{status} {filename}: {coverage_pct:.1f}%")
            
            print("\\n" + "-" * 40)
            print(f"Total files analyzed: {total_files}")
            print(f"Files with <90% coverage: {len(low_coverage_files)}")
            print(f"Files with >=90% coverage: {total_files - len(low_coverage_files)}")
            
            if low_coverage_files:
                print("\\n❌ Files with low coverage (<90%):")
                for filename, rate in low_coverage_files:
                    coverage_pct = rate * 100
                    print(f"   {filename}: {coverage_pct:.1f}%")
                print("\\n💡 Recommendation: Increase test coverage for these files.")
            
            overall_good = len(low_coverage_files) == 0
            print(f"\\n{'✅ ALL FILES MEET COVERAGE REQUIREMENTS!' if overall_good else '❌ SOME FILES NEED IMPROVED COVERAGE'}")
            
            return overall_good
            
        except Exception as e:
            print(f"⚠️  Could not parse coverage XML: {e}")
            # Fallback: just run coverage report to see
            subprocess.run([sys.executable, "-m", "coverage", "report"], check=True)


def verify_individual_file_coverage():
    """Verify that individual file coverage is >90%."""
    print("🔍 Running pytest-cov to check individual file coverage...")
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/", "--cov=pymars", "--cov-report=term-missing", 
        "--cov-report=html:htmlcov", "--cov-report=xml:coverage.xml", 
        "-v", "--tb=line"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print the output to see coverage per file
    print("Coverage Report Output:")
    print("-" * 40)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    # Parse the output to check if any file has <90% coverage
    lines = result.stdout.split('\\n')
    
    # Look for the line that contains 'TOTAL' 
    total_line = None
    for line in lines:
        if 'TOTAL' in line:
            total_line = line
            break
    
    if total_line:
        print(f"\\n📊 Total Coverage: {total_line}")
        # Look for individual file coverage information
        print("\\n📊 Individual File Coverage:")
        print("-" * 60)
        
        # Find lines with coverage information
        file_coverages = []
        for line in lines:
            if '.py' in line and 'pymars/' in line:
                parts = line.split()
                if len(parts) >= 4:
                    filename = parts[0] if parts[0].endswith('.py') else 'N/A'
                    try:
                        # Find the coverage percentage
                        for part in parts[1:]:
                            if part.endswith('%'):
                                coverage = float(part[:-1])
                                file_coverages.append((filename, coverage))
                                break
                    except:
                        pass
        
        # Check individual file coverage
        low_coverage_count = 0
        for filename, coverage in file_coverages:
            if coverage < 90.0:
                print(f"⚠️  {filename}: {coverage:.1f}%")
                low_coverage_count += 1
            else:
                print(f"✅ {filename}: {coverage:.1f}%")
        
        if low_coverage_count > 0:
            print(f"\\n❌ {low_coverage_count} files have <90% coverage")
            return False
        else:
            print(f"\\n✅ All {len(file_coverages)} files have >=90% coverage")
            return True
    else:
        print("❌ Could not find total coverage line in output")
        return False


if __name__ == "__main__":
    success = verify_individual_file_coverage()
    if success:
        print("\\n🎉 All individual files have >90% coverage! Coverage verification passed!")
    else:
        print("\\n❌ Some files have <90% coverage. Need to improve test coverage.")
    
    # Clean up coverage files
    try:
        os.remove("coverage.xml")
        os.system("rm -rf htmlcov/ .coverage .coverage.*")
    except:
        pass
    
    exit(0 if success else 1)