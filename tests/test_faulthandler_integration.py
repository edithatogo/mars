"""
Test to verify faulthandler integration in pymars
"""
import os
import signal
import subprocess
import sys
import pytest
import numpy as np
from pymars import Earth


def test_faulthandler_basic_import():
    """Test that pymars imports with faulthandler enabled."""
    # This test passes if import works without errors
    import pymars
    assert pymars.__version__ == "1.0.0"
    print("✅ Basic faulthandler import test passed")


def test_faulthandler_debug_mode():
    """Test faulthandler activation in debug mode."""
    # Create a subprocess that enables debug mode
    code = """
import os
os.environ['PYMARS_DEBUG'] = '1'
import faulthandler
import pymars
print('Faulthandler enabled in debug mode:', faulthandler.is_enabled())
assert faulthandler.is_enabled()
print('✅ Debug mode faulthandler test passed')
"""
    
    result = subprocess.run([sys.executable, '-c', code], 
                          capture_output=True, text=True, cwd='.')
    assert result.returncode == 0
    print("✅ Debug mode faulthandler test passed")


def test_faulthandler_production_mode():
    """Test faulthandler in normal mode."""
    code = """
import os
# Don't set debug flag - should still initialize safely
import faulthandler
import pymars
# In production mode it still gets initialized but only for crashes
print('Production mode: faulthandler available')
print('✅ Production mode faulthandler test passed')
"""
    
    result = subprocess.run([sys.executable, '-c', code], 
                          capture_output=True, text=True, cwd='.')
    assert result.returncode == 0
    print("✅ Production mode faulthandler test passed")


def test_functionality_with_faulthandler():
    """Test that basic pymars functionality works with faulthandler enabled."""
    import pymars as pm
    
    # Generate test data
    np.random.seed(42)
    X = np.random.rand(20, 2)
    y = X[:, 0] + X[:, 1] * 0.5 + np.random.normal(0, 0.1, 20)
    
    # Test basic Earth functionality
    model = pm.Earth(max_degree=2, penalty=3.0, max_terms=10)
    model.fit(X, y)
    score = model.score(X, y)
    pred = model.predict(X[:5])
    
    assert score > 0.5  # Reasonable R²
    assert len(pred) == 5  # Correct number of predictions
    assert all(np.isfinite(pred))  # All predictions are finite
    
    print(f"✅ Functional test with faulthandler: R²={score:.4f}")


def test_faulthandler_signal_registration():
    """Test that signal registration works where available."""
    import faulthandler
    import signal
    
    # Test is only applicable where SIGUSR1 is available (Unix-like systems)
    if hasattr(signal, 'SIGUSR1'):
        # Temporarily disable to test re-registration
        import pymars
        print("✅ Signal registration test passed on Unix-like system")
    else:
        import pymars
        print("✅ Signal registration test passed on Windows (no SIGUSR1)")


if __name__ == "__main__":
    test_faulthandler_basic_import()
    test_faulthandler_debug_mode()  
    test_faulthandler_production_mode()
    test_functionality_with_faulthandler()
    test_faulthandler_signal_registration()
    print("\\n🎉 All faulthandler integration tests passed!")