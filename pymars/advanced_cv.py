"""
Advanced cross-validation strategies for pymars.

This module provides advanced cross-validation strategies beyond the basic
scikit-learn compatibility to improve model selection and evaluation.
"""
import logging
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
from sklearn.model_selection import (
    KFold, 
    StratifiedKFold, 
    TimeSeriesSplit,
    cross_val_score,
    cross_validate
)
from sklearn.metrics import make_scorer
from . import EarthRegressor, EarthClassifier
from .cv import EarthCV

logger = logging.getLogger(__name__)


class AdvancedEarthCV(EarthCV):
    """
    Advanced cross-validation helper for Earth models with enhanced strategies.
    
    This class extends EarthCV with advanced cross-validation strategies
    for better model selection and evaluation.
    """
    
    def __init__(self, estimator, cv=None, scoring=None, 
                 advanced_cv_strategy: str = 'standard',
                 n_jobs: Optional[int] = None, 
                 verbose: int = 0, 
                 pre_dispatch: Union[str, int] = '2*n_jobs'):
        """
        Initialize AdvancedEarthCV.
        
        Parameters
        ----------
        estimator : estimator object
            An EarthRegressor or EarthClassifier estimator.
        cv : int, cross-validation generator or iterable, optional (default=None)
            Determines the cross-validation splitting strategy.
        scoring : str, callable, list/tuple, or dict, optional (default=None)
            A single str or a callable to evaluate the predictions.
        advanced_cv_strategy : str, optional (default='standard')
            Advanced cross-validation strategy to use. Options:
            - 'standard': Basic K-Fold cross-validation
            - 'stratified': Stratified K-Fold for classification
            - 'timeseries': Time series split for temporal data
            - 'nested': Nested cross-validation for hyperparameter tuning
            - 'bootstrap': Bootstrap sampling for robust estimates
            - 'monte_carlo': Monte Carlo cross-validation with random splits
        n_jobs : int, optional (default=None)
            Number of jobs to run in parallel.
        verbose : int, optional (default=0)
            Controls the verbosity.
        pre_dispatch : int or str, optional (default='2*n_jobs')
            Controls the number of jobs that get dispatched during parallel execution.
        """
        super().__init__(estimator, cv=cv, scoring=scoring)
        self.advanced_cv_strategy = advanced_cv_strategy
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
    
    def _get_cv_strategy(self, y=None):
        """
        Get the appropriate cross-validation strategy based on the advanced_cv_strategy parameter.
        
        Parameters
        ----------
        y : array-like, optional
            Target values (used for stratified CV).
            
        Returns
        -------
        cv_strategy : cross-validation generator
            The selected cross-validation strategy.
        """
        if self.cv is not None:
            return self.cv
        
        if self.advanced_cv_strategy == 'standard':
            return KFold(n_splits=5, shuffle=True, random_state=42)
        elif self.advanced_cv_strategy == 'stratified':
            if y is not None:
                # Check if y is categorical for stratification
                unique_vals = np.unique(y)
                if len(unique_vals) <= 10:  # Assume categorical if <= 10 unique values
                    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            # Fallback to standard K-Fold
            return KFold(n_splits=5, shuffle=True, random_state=42)
        elif self.advanced_cv_strategy == 'timeseries':
            return TimeSeriesSplit(n_splits=5)
        elif self.advanced_cv_strategy == 'bootstrap':
            from sklearn.model_selection import ShuffleSplit
            return ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
        elif self.advanced_cv_strategy == 'monte_carlo':
            from sklearn.model_selection import ShuffleSplit
            return ShuffleSplit(n_splits=20, test_size=0.2, random_state=42)
        else:
            # Default to standard K-Fold
            return KFold(n_splits=5, shuffle=True, random_state=42)
    
    def advanced_score(self, X, y):
        """
        Score the model using advanced cross-validation strategies.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        scores : numpy.ndarray of shape (n_splits,)
            Array of scores of the estimator for each cv split.
        """
        # Get appropriate CV strategy
        cv_strategy = self._get_cv_strategy(y)
        
        # Perform cross-validation
        scores = cross_val_score(
            self.estimator, 
            X, 
            y, 
            cv=cv_strategy, 
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            pre_dispatch=self.pre_dispatch
        )
        
        return scores
    
    def nested_cv_score(self, X, y, param_grid=None):
        """
        Perform nested cross-validation for hyperparameter tuning and model evaluation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.
        param_grid : dict, optional (default=None)
            Parameter grid for hyperparameter tuning. If None, uses default grid.
            
        Returns
        -------
        scores : numpy.ndarray of shape (n_outer_splits,)
            Array of scores from the outer CV loop.
        best_params : list of dicts
            Best parameters found in each inner CV loop.
        """
        from sklearn.model_selection import GridSearchCV
        
        if param_grid is None:
            # Default parameter grid
            param_grid = {
                'max_degree': [1, 2],
                'penalty': [2.0, 3.0, 4.0],
                'max_terms': [10, 15, 20]
            }
        
        # Outer CV strategy
        outer_cv = self._get_cv_strategy(y)
        
        # Inner CV strategy (different random state)
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=43)
        
        # Create GridSearchCV object
        grid_search = GridSearchCV(
            self.estimator,
            param_grid,
            cv=inner_cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        
        # Perform nested cross-validation
        nested_scores = cross_val_score(
            grid_search, 
            X, 
            y, 
            cv=outer_cv, 
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            pre_dispatch=self.pre_dispatch
        )
        
        # Get best parameters from each fold (this requires a more complex approach)
        best_params_list = []
        
        # For each outer fold, fit a GridSearchCV and collect best params
        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            grid_search.fit(X_train, y_train)
            best_params_list.append(grid_search.best_params_)
        
        return nested_scores, best_params_list


class BootstrapEarthCV(EarthCV):
    """
    Bootstrap cross-validation for Earth models.
    
    This class provides bootstrap sampling for robust model evaluation.
    """
    
    def __init__(self, estimator, n_bootstraps: int = 100, 
                 bootstrap_size: float = 0.8, random_state: int = 42):
        """
        Initialize BootstrapEarthCV.
        
        Parameters
        ----------
        estimator : estimator object
            An EarthRegressor or EarthClassifier estimator.
        n_bootstraps : int, optional (default=100)
            Number of bootstrap samples to generate.
        bootstrap_size : float, optional (default=0.8)
            Fraction of samples to include in each bootstrap sample.
        random_state : int, optional (default=42)
            Random state for reproducibility.
        """
        super().__init__(estimator)
        self.n_bootstraps = n_bootstraps
        self.bootstrap_size = bootstrap_size
        self.random_state = random_state
    
    def bootstrap_score(self, X, y):
        """
        Score the model using bootstrap sampling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        scores : numpy.ndarray of shape (n_bootstraps,)
            Array of scores from bootstrap samples.
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        bootstrap_scores = []
        
        for i in range(self.n_bootstraps):
            # Generate bootstrap sample
            bootstrap_indices = np.random.choice(
                n_samples, 
                size=int(n_samples * self.bootstrap_size), 
                replace=True
            )
            
            # Split into train/test
            test_indices = np.setdiff1d(np.arange(n_samples), bootstrap_indices)
            
            if len(test_indices) == 0:
                # All samples were selected, use a small random test set
                test_indices = np.random.choice(n_samples, size=max(1, int(n_samples * 0.2)), replace=False)
                bootstrap_indices = np.setdiff1d(bootstrap_indices, test_indices)
            
            if len(bootstrap_indices) == 0 or len(test_indices) == 0:
                continue
                
            X_train = X[bootstrap_indices]
            y_train = y[bootstrap_indices]
            X_test = X[test_indices]
            y_test = y[test_indices]
            
            # Fit and score
            estimator_copy = type(self.estimator)(**self.estimator.get_params())
            estimator_copy.fit(X_train, y_train)
            score = estimator_copy.score(X_test, y_test)
            bootstrap_scores.append(score)
        
        return np.array(bootstrap_scores)


def demo_advanced_cv():
    """Demonstrate advanced cross-validation strategies."""
    print("Demonstrating advanced cross-validation strategies...")
    
    # Generate test data
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = X[:, 0] + X[:, 1] * 0.5 + np.sin(X[:, 2] * np.pi) + np.random.normal(0, 0.1, 100)
    y_class = (y > np.median(y)).astype(int)
    
    print("\\n1. AdvancedEarthCV with different strategies:")
    
    # Test standard CV
    model_reg = EarthRegressor(max_degree=2, penalty=3.0, max_terms=15)
    cv_standard = AdvancedEarthCV(model_reg, advanced_cv_strategy='standard')
    scores_standard = cv_standard.advanced_score(X, y)
    print(f"   Standard CV scores: {[f'{s:.4f}' for s in scores_standard]}")
    print(f"   Mean: {np.mean(scores_standard):.4f} ± {np.std(scores_standard):.4f}")
    
    # Test stratified CV
    cv_stratified = AdvancedEarthCV(model_reg, advanced_cv_strategy='stratified')
    scores_stratified = cv_stratified.advanced_score(X, y)
    print(f"   Stratified CV scores: {[f'{s:.4f}' for s in scores_stratified]}")
    print(f"   Mean: {np.mean(scores_stratified):.4f} ± {np.std(scores_stratified):.4f}")
    
    # Test bootstrap CV
    cv_bootstrap = BootstrapEarthCV(model_reg, n_bootstraps=20, bootstrap_size=0.8)
    scores_bootstrap = cv_bootstrap.bootstrap_score(X, y)
    print(f"   Bootstrap CV scores: {[f'{s:.4f}' for s in scores_bootstrap[:10]]}...")
    print(f"   Mean: {np.mean(scores_bootstrap):.4f} ± {np.std(scores_bootstrap):.4f}")
    
    # Test Monte Carlo CV
    cv_monte_carlo = AdvancedEarthCV(model_reg, advanced_cv_strategy='monte_carlo')
    scores_mc = cv_monte_carlo.advanced_score(X, y)
    print(f"   Monte Carlo CV scores: {[f'{s:.4f}' for s in scores_mc]}")
    print(f"   Mean: {np.mean(scores_mc):.4f} ± {np.std(scores_mc):.4f}")
    
    print("\\n2. Nested cross-validation:")
    # Test nested CV
    try:
        param_grid = {
            'max_degree': [1, 2],
            'penalty': [2.0, 3.0],
            'max_terms': [10, 15]
        }
        nested_scores, best_params = cv_standard.nested_cv_score(X, y, param_grid)
        print(f"   Nested CV scores: {[f'{s:.4f}' for s in nested_scores]}")
        print(f"   Mean: {np.mean(nested_scores):.4f} ± {np.std(nested_scores):.4f}")
        print(f"   Best params from first fold: {best_params[0] if best_params else 'None'}")
        print("   ✅ Nested CV completed successfully")
    except Exception as e:
        print(f"   ⚠️  Nested CV failed: {e}")
    
    print("\\nAdvanced cross-validation demonstration completed!")


if __name__ == "__main__":
    demo_advanced_cv()