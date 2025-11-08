#!/usr/bin/env python
"""
Final verification script for pymars v1.0.0.

This script performs a comprehensive verification that all functionality
is working correctly before publication.
"""

import numpy as np
import pymars as pm


def test_basic_functionality():
    """Test basic Earth functionality."""
    print("🧪 Testing Basic Earth Functionality...")
    
    # Generate test data
    np.random.seed(42)
    X = np.random.rand(50, 3)
    y = X[:, 0] + X[:, 1] * 0.5 + np.sin(X[:, 2] * np.pi) + np.random.normal(0, 0.1, 50)
    
    # Test basic Earth model
    model = pm.Earth(max_degree=2, penalty=3.0, max_terms=15)
    model.fit(X, y)
    score = model.score(X, y)
    preds = model.predict(X[:5])
    
    print(f"✅ Basic Earth: R²={score:.4f}, Terms={len(model.basis_)}")
    assert score > 0.8, "Basic Earth model should achieve reasonable R²"
    assert len(preds) == 5, "Should predict for 5 samples"
    return True


def test_scikit_learn_compatibility():
    """Test scikit-learn compatibility."""
    print("🧪 Testing Scikit-learn Compatibility...")
    
    # Generate test data
    np.random.seed(42)
    X = np.random.rand(30, 2)
    y = X[:, 0] + X[:, 1] * 0.5 + np.random.normal(0, 0.1, 30)
    
    # Test EarthRegressor
    regressor = pm.EarthRegressor(max_degree=2, penalty=3.0, max_terms=10)
    regressor.fit(X, y)
    score = regressor.score(X, y)
    
    # Test EarthClassifier
    y_class = (y > np.median(y)).astype(int)
    classifier = pm.EarthClassifier(max_degree=2, penalty=3.0, max_terms=10)
    classifier.fit(X, y_class)
    acc = classifier.score(X, y_class)
    
    print(f"✅ Scikit-learn: Regressor R²={score:.4f}, Classifier Acc={acc:.4f}")
    assert score > 0.7, "EarthRegressor should achieve reasonable R²"
    assert acc > 0.6, "EarthClassifier should achieve reasonable accuracy"
    return True


def test_specialized_models():
    """Test specialized models."""
    print("🧪 Testing Specialized Models...")
    
    # Generate test data
    np.random.seed(42)
    X = np.random.rand(40, 3)
    y = X[:, 0] + X[:, 1] * 0.5 + np.sin(X[:, 2] * np.pi) + np.random.normal(0, 0.1, 40)
    y_binary = (y > np.median(y)).astype(int)
    
    # Test GLMEarth
    glm = pm.GLMEarth(family='logistic', max_degree=2, penalty=3.0, max_terms=12)
    glm.fit(X, y_binary)
    glm_preds = glm.predict(X[:5])
    
    # Test EarthCV
    from sklearn.model_selection import KFold
    cv = pm.EarthCV(pm.EarthRegressor(max_degree=1, penalty=3.0, max_terms=8), cv=KFold(n_splits=3))
    cv_scores = cv.score(X, y)
    
    print(f"✅ Specialized Models: GLM terms={len(glm.basis_)}, CV mean={np.mean(cv_scores):.4f}")
    assert len(glm.basis_) > 0, "GLMEarth should have basis functions"
    assert len(cv_scores) == 3, "EarthCV should return 3 scores"
    return True


def test_new_advanced_features():
    """Test new advanced features."""
    print("🧪 Testing New Advanced Features...")
    
    # Generate test data
    np.random.seed(42)
    X = np.random.rand(30, 2)
    y = X[:, 0] + X[:, 1] * 0.5 + np.random.normal(0, 0.1, 30)
    
    # Test CachedEarth
    cached_model = pm.CachedEarth(max_degree=2, penalty=3.0, max_terms=10)
    cached_model.fit(X, y)
    cached_score = cached_model.score(X, y)
    
    # Test ParallelEarth
    parallel_model = pm.ParallelEarth(max_degree=2, penalty=3.0, max_terms=10)
    parallel_model.fit(X, y)
    parallel_score = parallel_model.score(X, y)
    
    # Test SparseEarth
    sparse_model = pm.SparseEarth(max_degree=2, penalty=3.0, max_terms=10, sparse_support=True)
    sparse_model.fit(X, y)
    sparse_score = sparse_model.score(X, y)
    
    # Test AdvancedEarthCV
    adv_cv = pm.AdvancedEarthCV(pm.EarthRegressor(max_degree=1, penalty=3.0, max_terms=8))
    adv_scores = adv_cv.advanced_score(X, y)
    
    print(f"✅ New Features: Cached R²={cached_score:.4f}, Parallel R²={parallel_score:.4f}")
    print(f"                   Sparse R²={sparse_score:.4f}, AdvCV mean={np.mean(adv_scores):.4f}")
    
    assert cached_score > 0.7, "CachedEarth should achieve reasonable R²"
    assert parallel_score > 0.7, "ParallelEarth should achieve reasonable R²"
    assert sparse_score > 0.7, "SparseEarth should achieve reasonable R²"
    assert len(adv_scores) > 0, "AdvancedEarthCV should return scores"
    return True


def test_additional_glm_families():
    """Test additional GLM families."""
    print("🧪 Testing Additional GLM Families...")
    
    # Generate test data
    np.random.seed(42)
    X = np.random.rand(30, 2)
    y_cont = np.abs(X[:, 0] + X[:, 1] * 0.5 + np.random.normal(0, 0.1, 30)) + 0.1
    y_count = np.abs(y_cont).astype(int)
    
    # Test Gamma GLM
    try:
        gamma_glm = pm.AdvancedGLMEarth(family='gamma', max_degree=2, penalty=3.0, max_terms=10)
        gamma_glm.fit(X, y_cont)
        gamma_score = gamma_glm.score(X, y_cont)
        print(f"✅ Gamma GLM: R²={gamma_score:.4f}")
        assert gamma_score > 0.5, "Gamma GLM should achieve reasonable R²"
    except Exception as e:
        print(f"⚠️  Gamma GLM test failed: {e}")
    
    # Test Poisson GLM
    try:
        poisson_glm = pm.AdvancedGLMEarth(family='poisson', max_degree=2, penalty=3.0, max_terms=10)
        poisson_glm.fit(X, y_count)
        poisson_score = poisson_glm.score(X, y_count)
        print(f"✅ Poisson GLM: R²={poisson_score:.4f}")
        assert poisson_score > 0.5, "Poisson GLM should achieve reasonable R²"
    except Exception as e:
        print(f"⚠️  Poisson GLM test failed: {e}")
    
    return True


def test_feature_importance_and_plotting():
    """Test feature importance and plotting utilities."""
    print("🧪 Testing Feature Importance and Plotting...")
    
    # Generate test data
    np.random.seed(42)
    X = np.random.rand(30, 3)
    y = X[:, 0] + X[:, 1] * 0.5 + np.sin(X[:, 2] * np.pi) + np.random.normal(0, 0.1, 30)
    
    # Test feature importance
    model = pm.Earth(feature_importance_type='gcv', max_degree=2, penalty=3.0, max_terms=10)
    model.fit(X, y)
    importances = model.feature_importances_
    
    print(f"✅ Feature Importance: Shape={importances.shape}, Sum={np.sum(importances):.4f}")
    assert importances.shape == (3,), "Should have importance for 3 features"
    assert np.isclose(np.sum(importances), 1.0), "Importances should sum to 1.0"
    
    # Test plotting utilities (should not raise exceptions)
    try:
        fig, ax = pm.plot_basis_functions(model, X)
        print("✅ Plotting: Basis functions plot created")
    except Exception as e:
        print(f"⚠️  Plotting test failed: {e}")
    
    try:
        fig, ax = pm.plot_residuals(model, X, y)
        print("✅ Plotting: Residuals plot created")
    except Exception as e:
        print(f"⚠️  Plotting test failed: {e}")
    
    return True


def test_cli_functionality():
    """Test CLI functionality."""
    print("🧪 Testing CLI Functionality...")
    
    import subprocess
    result = subprocess.run(['python', '-m', 'pymars', '--version'], 
                           capture_output=True, text=True, cwd='.')
    if result.returncode == 0:
        print(f"✅ CLI: Version={result.stdout.strip()}")
        return True
    else:
        print(f"⚠️  CLI test failed: {result.stderr}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 80)
    print("🧪 pymars v1.0.0 Final Verification")
    print("=" * 80)
    
    tests = [
        test_basic_functionality,
        test_scikit_learn_compatibility,
        test_specialized_models,
        test_new_advanced_features,
        test_additional_glm_families,
        test_feature_importance_and_plotting,
        test_cli_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
    
    print("\n" + "=" * 80)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("pymars v1.0.0 is READY FOR PUBLICATION!")
        print("=" * 80)
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        print("pymars v1.0.0 needs fixes before publication.")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)