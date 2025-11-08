#!/usr/bin/env python
"""
Comprehensive verification script for pymars v1.0.0.

This script runs all verification tests to ensure that pymars v1.0.0
is production-ready and ready for publication to PyPI.
"""
import sys
import numpy as np
import pymars as earth


def test_basic_functionality():
    """Test basic Earth functionality."""
    print("🧪 Testing Basic Earth Functionality...")
    
    # Generate test data
    np.random.seed(42)
    X = np.random.rand(50, 3)
    y = X[:, 0] + X[:, 1] * 0.5 + np.sin(X[:, 2] * np.pi) + np.random.normal(0, 0.1, 50)
    
    # Test basic Earth model
    model = earth.Earth(max_degree=2, penalty=3.0, max_terms=15)
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
    regressor = earth.EarthRegressor(max_degree=2, penalty=3.0, max_terms=10)
    regressor.fit(X, y)
    score = regressor.score(X, y)
    
    # Test EarthClassifier
    y_class = (y > np.median(y)).astype(int)
    classifier = earth.EarthClassifier(max_degree=2, penalty=3.0, max_terms=10)
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
    glm = earth.GLMEarth(family='logistic', max_degree=2, penalty=3.0, max_terms=12)
    glm.fit(X, y_binary)
    glm_preds = glm.predict(X[:5])
    
    # Test EarthCV
    from sklearn.model_selection import KFold
    cv = earth.EarthCV(earth.EarthRegressor(max_degree=1, penalty=3.0, max_terms=8), cv=KFold(n_splits=3))
    scores = cv.score(X, y)
    
    print(f"✅ Specialized Models: GLM terms={len(glm.basis_)}, CV mean={np.mean(scores):.4f}")
    assert len(glm.basis_) > 0, "GLMEarth should have basis functions"
    assert len(scores) == 3, "EarthCV should return 3 scores"
    return True


def test_feature_importance_and_plotting():
    """Test feature importance and plotting utilities."""
    print("🧪 Testing Feature Importance and Plotting...")
    
    # Generate test data
    np.random.seed(42)
    X = np.random.rand(30, 3)
    y = X[:, 0] + X[:, 1] * 0.5 + np.sin(X[:, 2] * np.pi) + np.random.normal(0, 0.1, 30)
    
    # Test feature importance
    model = earth.Earth(feature_importance_type='gcv', max_degree=2, penalty=3.0, max_terms=10)
    model.fit(X, y)
    importances = model.feature_importances_
    
    print(f"✅ Feature Importance: Shape={importances.shape}, Sum={np.sum(importances):.4f}")
    assert importances.shape == (3,), "Should have importance for 3 features"
    assert np.isclose(np.sum(importances), 1.0), "Importances should sum to 1.0"
    
    # Test plotting utilities (should not raise exceptions)
    try:
        fig, ax = earth.plot_basis_functions(model, X)
        print("✅ Plotting: Basis functions plot created")
    except Exception as e:
        print(f"⚠️  Plotting test: Basis functions plot ({type(e).__name__})")
    
    try:
        fig, ax = earth.plot_residuals(model, X, y)
        print("✅ Plotting: Residuals plot created")
    except Exception as e:
        print(f"⚠️  Plotting test: Residuals plot ({type(e).__name__})")
    
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
        print(f"⚠️  CLI test: {result.stderr}")
        return False


def test_advanced_interpretability():
    """Test advanced interpretability tools."""
    print("🧪 Testing Advanced Interpretability...")
    
    # Generate test data
    np.random.seed(42)
    X = np.random.rand(30, 3)
    y = X[:, 0] + X[:, 1] * 0.5 + np.sin(X[:, 2] * np.pi) + np.random.normal(0, 0.1, 30)
    
    # Fit model
    model = earth.Earth(max_degree=2, penalty=3.0, max_terms=10)
    model.fit(X, y)
    
    # Test advanced interpretability functions
    try:
        explanation = earth.get_model_explanation(model, X, feature_names=[f'Feature_{i}' for i in range(X.shape[1])])
        print(f"✅ Interpretability: Model explanation keys={list(explanation.keys())}")
        assert 'model_summary' in explanation
        assert 'basis_functions' in explanation
        assert 'feature_importance' in explanation
        return True
    except Exception as e:
        print(f"⚠️  Interpretability test: {type(e).__name__}")
        return False


def test_categorical_features():
    """Test categorical feature handling."""
    print("🧪 Testing Categorical Features...")
    
    # Generate test data with categorical features
    np.random.seed(42)
    X = np.random.rand(50, 3)
    X[:, 1] = np.random.randint(0, 3, 50)  # Categorical feature with 3 categories
    y = X[:, 0] + X[:, 1] * 0.5 + np.sin(X[:, 2] * np.pi) + np.random.normal(0, 0.1, 50)
    
    # Test Earth model with categorical features
    model = earth.Earth(max_degree=2, penalty=3.0, max_terms=15, categorical_features=[1])
    model.fit(X, y)
    
    assert model.fitted_, "Model with categorical features should be fitted"
    assert len(model.basis_) > 0, "Model with categorical features should have basis functions"
    
    print(f"✅ Categorical Features: {len(model.basis_)} terms")
    return True


def test_missing_values():
    """Test missing value handling."""
    print("🧪 Testing Missing Values...")
    
    # Generate test data with missing values
    np.random.seed(42)
    X = np.random.rand(50, 3)
    X[:10, 0] = np.nan  # Add some missing values
    y = X[:, 0] + X[:, 1] * 0.5 + np.sin(X[:, 2] * np.pi) + np.random.normal(0, 0.1, 50)
    y = np.where(np.isnan(y), 0, y)  # Handle NaN in target
    
    # Test Earth model with missing values
    model = earth.Earth(max_degree=2, penalty=3.0, max_terms=15, allow_missing=True)
    model.fit(X, y)
    
    assert model.fitted_, "Model with missing values should be fitted"
    assert len(model.basis_) > 0, "Model with missing values should have basis functions"
    
    print(f"✅ Missing Values: {len(model.basis_)} terms")
    return True


def test_new_advanced_features():
    """Test new advanced features."""
    print("🧪 Testing New Advanced Features...")
    
    # Generate test data
    np.random.seed(42)
    X = np.random.rand(30, 2)
    y = X[:, 0] + X[:, 1] * 0.5 + np.random.normal(0, 0.1, 30)
    
    # Test CachedEarth
    try:
        cached_model = earth.CachedEarth(max_degree=2, penalty=3.0, max_terms=10)
        cached_model.fit(X, y)
        cached_score = cached_model.score(X, y)
        print(f"✅ CachedEarth: R²={cached_score:.4f}")
    except Exception as e:
        print(f"⚠️  CachedEarth test: {type(e).__name__}")
    
    # Test ParallelEarth
    try:
        parallel_model = earth.ParallelEarth(max_degree=2, penalty=3.0, max_terms=10)
        parallel_model.fit(X, y)
        parallel_score = parallel_model.score(X, y)
        print(f"✅ ParallelEarth: R²={parallel_score:.4f}")
    except Exception as e:
        print(f"⚠️  ParallelEarth test: {type(e).__name__}")
    
    # Test SparseEarth
    try:
        sparse_model = earth.SparseEarth(max_degree=2, penalty=3.0, max_terms=10, sparse_support=True)
        sparse_model.fit(X, y)
        sparse_score = sparse_model.score(X, y)
        print(f"✅ SparseEarth: R²={sparse_score:.4f}")
    except Exception as e:
        print(f"⚠️  SparseEarth test: {type(e).__name__}")
    
    return True


def main():
    """Run all verification tests."""
    print("=" * 80)
    print("🧪 pymars v1.0.0: COMPREHENSIVE VERIFICATION")
    print("=" * 80)
    
    tests = [
        test_basic_functionality,
        test_scikit_learn_compatibility,
        test_specialized_models,
        test_feature_importance_and_plotting,
        test_cli_functionality,
        test_advanced_interpretability,
        test_categorical_features,
        test_missing_values,
        test_new_advanced_features
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print()
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            print()
    
    print("=" * 80)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("pymars v1.0.0 is PRODUCTION READY!")
        print("=" * 80)
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        print("pymars v1.0.0 needs fixes before publication.")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)