"""
Enhanced typed interface for pymars Earth class with protocol definitions for better type checking.
"""
from abc import abstractmethod
from typing import Protocol, TypeVar, List, Optional, Union, Tuple, Any, Dict, Sequence, Literal
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


# Define protocols for better interface contracts
class BasisFunctionProtocol(Protocol):
    """Protocol defining the interface for basis functions."""
    
    def transform(self, X: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """Transform input data X using the basis function."""
        ...
    
    def get_involved_variables(self) -> List[int]:
        """Get the indices of variables involved in this basis function."""
        ...
    
    def is_constant(self) -> bool:
        """Check if this is a constant basis function."""
        ...
    
    def degree(self) -> int:
        """Get the degree of this basis function."""
        ...


class FittableModelProtocol(Protocol):
    """Protocol for models that can be fitted."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FittableModelProtocol':
        """Fit the model to training data."""
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        ...
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate the model's performance score."""
        ...


# Import Earth here to avoid circular imports issues
from .earth import Earth

# Type variable for model instances
ModelT = TypeVar('ModelT', bound='TypedEarth')


class TypedEarth(BaseEstimator, RegressorMixin):
    """Typed version of Earth with enhanced type annotations for better static analysis.
    
    This class provides the same functionality as Earth but with enhanced type annotations
    to support better static type checking and developer experience.
    """
    
    def __init__(
        self,
        max_degree: int = 1,
        penalty: float = 3.0,
        max_terms: Optional[int] = None,
        minspan_alpha: float = 0.0,
        endspan_alpha: float = 0.0,
        allow_linear: bool = True,
        allow_missing: bool = False,
        feature_importance_type: Optional[str] = None,
        categorical_features: Optional[List[int]] = None,
        minspan: int = -1,
        endspan: int = -1,
    ) -> None:
        """Initialize the Earth model with enhanced type annotations.
        
        Args:
            max_degree: Maximum degree of interaction terms
            penalty: Penalty parameter for GCV criterion
            max_terms: Maximum number of basis functions
            minspan_alpha: Minimum span as fraction of range for knot selection
            endspan_alpha: Minimum span at edges as fraction of range
            allow_linear: Whether to allow linear basis functions
            allow_missing: Whether to allow missing values in X
            feature_importance_type: Type of feature importance calculation
            categorical_features: List of categorical feature indices
            minspan: Minimum span for knot selection (-1 uses minspan_alpha)
            endspan: End span for knot selection (-1 uses endspan_alpha)
        """
        self.max_degree = max_degree
        self.penalty = penalty
        self.max_terms = max_terms
        self.minspan_alpha = minspan_alpha
        self.endspan_alpha = endspan_alpha
        self.minspan = minspan
        self.endspan = endspan
        self.allow_linear = allow_linear
        self.allow_missing = allow_missing
        self.feature_importance_type = feature_importance_type
        self.categorical_features = categorical_features

        # Attributes that will be learned during fit
        self.basis_: Optional[List[BasisFunctionProtocol]] = None
        self.coef_: Optional[np.ndarray] = None
        self.record_ = None

        self.rss_: Optional[float] = None
        self.mse_: Optional[float] = None
        self.gcv_: Optional[float] = None

        self.feature_importances_: Optional[np.ndarray] = None

        self.fitted_: bool = False
        self.categorical_imputer_ = None
        
        # Store for later use in model
        self.X_original_: Optional[np.ndarray] = None
        self.missing_mask_: Optional[np.ndarray] = None
    
    @abstractmethod
    def fit(self, X: XType, y: YType) -> 'TypedEarth':
        """Fit the Earth model to training data with typed interface.
        
        Args:
            X: Training input samples of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: Fitted model instance
        """
        pass
    
    @abstractmethod
    def predict(self, X: XType) -> np.ndarray:
        """Predict target values for samples in X.
        
        Args:
            X: Input samples of shape (n_samples, n_features)
            
        Returns:
            Predicted target values of shape (n_samples,)
        """
        pass
    
    @abstractmethod
    def score(self, X: XType, y: YType) -> float:
        """Return the coefficient of determination R^2 of the prediction.
        
        Args:
            X: Input samples of shape (n_samples, n_features)
            y: True target values of shape (n_samples,)
            
        Returns:
            R^2 score of the prediction
        """
        pass
    
    @abstractmethod
    def summary(self) -> str:
        """Return a summary of the fitted model.
        
        Returns:
            String summary of the model
        """
        pass
    
    @abstractmethod
    def _build_basis_matrix(self, X_processed: np.ndarray, basis_functions: List[BasisFunctionProtocol], 
                           missing_mask: np.ndarray) -> np.ndarray:
        """Build the basis matrix from processed input data and basis functions.
        
        Args:
            X_processed: Processed input data
            basis_functions: List of basis functions to apply
            missing_mask: Boolean mask indicating missing values
            
        Returns:
            Basis matrix with shape (n_samples, n_basis_functions)
        """
        pass
    
    @abstractmethod
    def _scrub_input_data(self, X: XType, y: YType) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate and preprocess input data.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Tuple of (processed_X, missing_mask, processed_y)
        """
        pass
    
    @abstractmethod
    def _set_fallback_model(self, X_processed: np.ndarray, y_processed: np.ndarray, 
                           missing_mask: np.ndarray, 
                           pruning_passer_instance_for_gcv_calc: Any) -> None:
        """Set a fallback model if the main model fitting fails.
        
        Args:
            X_processed: Processed input data
            y_processed: Processed target values
            missing_mask: Missing value mask
            pruning_passer_instance_for_gcv_calc: Pruning passer instance for GCV calculation
        """
        pass
    
    @abstractmethod
    def summary_feature_importances(self, sort_by_importance: bool = True) -> str:
        """Return a summary of feature importances.
        
        Args:
            sort_by_importance: Whether to sort features by importance
            
        Returns:
            Summary string of feature importances
        """
        pass


# Define type aliases for common types used in pymars
FeatureImportanceType = Literal['nb_subsets', 'gcv', 'rss', None]
XType = Union[np.ndarray, Sequence]
YType = Union[np.ndarray, Sequence]

# Export the interface for better static analysis
__all__ = [
    'TypedEarth', 
    'BasisFunctionProtocol', 
    'FittableModelProtocol',
    'FeatureImportanceType',
    'XType',
    'YType'
]