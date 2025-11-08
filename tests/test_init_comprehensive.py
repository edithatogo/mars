"""
Comprehensive tests for pymars __init__.py to achieve >95% coverage
"""
import pytest
import sys
import os
from unittest import mock


def test_init_module_imports():
    """Test all imports in __init__.py work correctly."""
    # This should trigger the init file to run
    import pymars as pm
    
    # Verify all expected classes are available
    assert hasattr(pm, 'Earth')
    assert hasattr(pm, 'EarthRegressor')
    assert hasattr(pm, 'EarthClassifier')
    assert hasattr(pm, 'GLMEarth')
    assert hasattr(pm, 'EarthCV')
    assert hasattr(pm, 'CategoricalImputer')
    
    # Test with enhanced classes
    assert hasattr(pm, 'CachedEarth')
    assert hasattr(pm, 'ParallelEarth')
    assert hasattr(pm, 'SparseEarth')
    
    print("✅ All core imports work correctly")


def test_faulthandler_activation_paths():
    """Test different faulthandler activation paths."""
    # Test normal import (not in debug mode)
    with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch.object(sys, 'argv', ['pymars']):
            import importlib
            # Reload module to test the path
            if 'pymars' in sys.modules:
                del sys.modules['pymars']
            
            import pymars as pm_norm
            # Should import without issues
            assert pm_norm.__version__ == "1.0.0"


def test_faulthandler_debug_mode():
    """Test faulthandler activation in debug mode."""
    # Test with debug environment
    with mock.patch.dict(os.environ, {'PYMARS_DEBUG': '1'}, clear=True):
        with mock.patch.object(sys, 'argv', ['pymars']):
            import importlib
            
            # Reload module to test debug path
            if 'pymars' in sys.modules:
                del sys.modules['pymars']
            
            import pymars as pm_debug
            # Should still work in debug mode
            assert pm_debug.__version__ == "1.0.0"


def test_faulthandler_argv_debug():
    """Test faulthandler activation via argv."""
    # Test with debug flag in argv
    with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch.object(sys, 'argv', ['pymars', '--debug']):
            import importlib
            
            # Reload module to test argv path
            if 'pymars' in sys.modules:
                del sys.modules['pymars']
            
            import pymars as pm_argv
            # Should work with argv flag
            assert pm_argv.__version__ == "1.0.0"


if __name__ == "__main__":
    test_init_module_imports()
    test_faulthandler_activation_paths() 
    test_faulthandler_debug_mode()
    test_faulthandler_argv_debug()
    
    print("✅ All __init__.py tests completed successfully!")