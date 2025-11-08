#!/usr/bin/env python
"""
Memory profiling script for pymars using memory_profiler.

This script profiles memory usage of pymars components to identify optimization opportunities.
"""
import numpy as np
from memory_profiler import profile
from pymars import Earth


@profile
def memory_profile_earth_model():
    """Profile memory usage of Earth model fitting."""
    print("🔧 Memory profiling Earth model...")
    
    # Generate test data
    X = np.random.rand(100, 5)
    y = np.sin(X[:, 0]) + X[:, 1] * 0.5 + np.random.normal(0, 0.1, 100)
    
    # Create and fit model
    model = Earth(max_degree=2, penalty=3.0, max_terms=15)
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X[:10])
    
    return model, predictions


@profile
def memory_profile_large_dataset():
    """Profile memory usage with a larger dataset."""
    print("🔧 Memory profiling with larger dataset...")
    
    # Generate larger test data
    X = np.random.rand(1000, 10)
    y = (np.sin(X[:, 0]) + X[:, 1] * X[:, 2] + 
         np.cos(X[:, 3]) * X[:, 4] + np.random.normal(0, 0.1, 1000))
    
    # Create and fit model
    model = Earth(max_degree=3, penalty=3.0, max_terms=30)
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X[:100])
    
    return model, predictions


@profile
def memory_profile_cross_validation():
    """Profile memory usage during cross-validation."""
    print("🔧 Memory profiling cross-validation...")
    
    from sklearn.model_selection import KFold
    from pymars.cv import EarthCV
    
    # Generate test data
    X = np.random.rand(200, 5)
    y = np.sin(X[:, 0]) + X[:, 1] * 0.5 + np.random.normal(0, 0.1, 200)
    
    # Create and run cross-validation
    cv = EarthCV(Earth(max_degree=2, penalty=3.0, max_terms=15), cv=KFold(n_splits=5))
    scores = cv.score(X, y)
    
    return cv, scores


@profile
def memory_profile_glm():
    """Profile memory usage of GLM models."""
    print("🔧 Memory profiling GLM models...")
    
    # Generate test data for classification
    X = np.random.rand(150, 4)
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.normal(0, 0.1, 150)) > np.median(X[:, 0])
    y = y.astype(int)
    
    from pymars.glm import GLMEarth
    
    # Create and fit GLM model
    glm_model = GLMEarth(family='logistic', max_degree=2, penalty=3.0, max_terms=10)
    glm_model.fit(X, y)
    
    # Make predictions
    predictions = glm_model.predict(X[:20])
    probabilities = glm_model.predict_proba(X[:20])
    
    return glm_model, predictions, probabilities


@profile
def memory_profile_categorical_features():
    """Profile memory usage with categorical features."""
    print("🔧 Memory profiling categorical features...")
    
    from pymars._categorical import CategoricalImputer
    
    # Generate test data with categorical features
    X = np.random.rand(100, 3)
    X[:, 1] = np.random.randint(0, 3, 100)  # Categorical feature with 3 categories
    y = X[:, 0] + (X[:, 1] == 0) * 0.5 + (X[:, 1] == 1) * 1.0 + (X[:, 1] == 2) * 1.5 + np.random.normal(0, 0.1, 100)
    
    # Create and fit model with categorical features
    model = Earth(max_degree=2, penalty=3.0, max_terms=15, categorical_features=[1])
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X[:10])
    
    return model, predictions


@profile
def memory_profile_missing_values():
    """Profile memory usage with missing values."""
    print("🔧 Memory profiling missing values...")
    
    # Generate test data with missing values
    X = np.random.rand(100, 3)
    X[5:10, 0] = np.nan  # Add some missing values
    X[15:20, 1] = np.nan
    y = X[:, 0] + X[:, 1] * 0.5 + np.random.normal(0, 0.1, 100)
    
    # Create and fit model with missing value support
    model = Earth(max_degree=2, penalty=3.0, max_terms=15, allow_missing=True)
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X[:10])
    
    return model, predictions


def main():
    """Run all memory profiling tests."""
    print("=" * 80)
    print("🧠 MEMORY PROFILING FOR PYMARS")
    print("=" * 80)
    
    # Run all memory profiling tests
    model1, preds1 = memory_profile_earth_model()
    print(f"\n✅ Basic Earth model memory profiled")
    
    model2, preds2 = memory_profile_large_dataset()
    print(f"\n✅ Large dataset memory profiled")
    
    cv, scores = memory_profile_cross_validation()
    print(f"\n✅ Cross-validation memory profiled")
    
    glm, glm_preds, glm_probs = memory_profile_glm()
    print(f"\n✅ GLM model memory profiled")
    
    cat_model, cat_preds = memory_profile_categorical_features()
    print(f"\n✅ Categorical features memory profiled")
    
    missing_model, missing_preds = memory_profile_missing_values()
    print(f"\n✅ Missing values memory profiled")
    
    print("\n" + "=" * 80)
    print("🧠 Memory profiling completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()