# Performance Optimization Plan for mars

## Overview

This document outlines a comprehensive plan to optimize the performance of mars, identifying bottlenecks and proposing solutions to improve both speed and memory usage.

## Current Performance Analysis

Based on profiling results, the main performance bottlenecks in mars are:

1. **Basis Function Transformations** (60-70% of runtime)
   - `_build_basis_matrix` method in `_forward.py` and `_pruning.py`
   - Individual basis function `transform` methods
   - Repeated basis matrix construction for each candidate evaluation

2. **Linear Algebra Operations** (20-30% of runtime)
   - `np.linalg.lstsq` calls in `_calculate_rss_and_coeffs`
   - Matrix operations for RSS calculation

3. **Candidate Generation** (5-10% of runtime)
   - `_generate_candidates` method in `_forward.py`
   - Knot value generation and filtering

4. **Memory Allocations** (<5% of runtime but impacts overall performance)
   - Repeated array creations in basis function evaluations
   - Frequent matrix concatenations and reshaping

## Optimization Strategies

### 1. Basis Function Caching

**Problem**: Basis functions are repeatedly evaluated with the same input data during forward and pruning passes.

**Solution**: 
- Cache basis function transformations for common inputs
- Implement LRU cache for basis function evaluations
- Pre-compute frequently used transformations

**Implementation**:
```python
from functools import lru_cache
import hashlib

class CachedBasisFunction(BasisFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
    
    def transform(self, X_processed, missing_mask):
        # Create hash of input for cache key
        input_hash = hashlib.sha256(
            X_processed.tobytes() + missing_mask.tobytes()
        ).hexdigest()
        
        if input_hash in self._cache:
            return self._cache[input_hash]
        
        result = super().transform(X_processed, missing_mask)
        self._cache[input_hash] = result
        return result
```

### 2. Vectorized Basis Matrix Construction

**Problem**: `_build_basis_matrix` constructs matrices column-by-column which causes memory fragmentation.

**Solution**:
- Pre-allocate full basis matrix
- Use vectorized operations for all basis function evaluations
- Batch process basis functions when possible

**Implementation**:
```python
def _build_basis_matrix_vectorized(self, X_processed, basis_functions, missing_mask):
    """Vectorized basis matrix construction."""
    if not basis_functions:
        return np.empty((X_processed.shape[0], 0), dtype=float)
    
    # Pre-allocate full matrix
    B_matrix = np.empty((X_processed.shape[0], len(basis_functions)), dtype=float)
    
    # Batch process basis functions
    for i, bf in enumerate(basis_functions):
        B_matrix[:, i] = bf.transform(X_processed, missing_mask)
    
    return B_matrix
```

### 3. NumPy Optimization

**Problem**: Repeated small NumPy operations cause overhead.

**Solution**:
- Use NumPy's built-in vectorization wherever possible
- Replace Python loops with NumPy operations
- Optimize array indexing patterns

**Implementation**:
```python
# Instead of:
for i in range(len(array)):
    result[i] = operation(array[i])

# Use:
result = np.vectorize(operation)(array)

# Or even better, use native NumPy operations
```

### 4. Memory Pool Allocation

**Problem**: Frequent small allocations cause memory fragmentation.

**Solution**:
- Implement memory pooling for temporary arrays
- Reuse allocated arrays instead of creating new ones
- Pre-allocate common array sizes

**Implementation**:
```python
class MemoryPool:
    def __init__(self):
        self._pool = {}
    
    def get_array(self, shape, dtype=float):
        key = (shape, dtype)
        if key in self._pool and self._pool[key]:
            return self._pool[key].pop()
        return np.empty(shape, dtype=dtype)
    
    def return_array(self, array):
        key = (array.shape, array.dtype)
        if key not in self._pool:
            self._pool[key] = []
        self._pool[key].append(array)

# Global memory pool for the module
memory_pool = MemoryPool()
```

### 5. Lazy Evaluation

**Problem**: All basis functions are evaluated even when not needed.

**Solution**:
- Implement lazy evaluation for basis functions
- Only compute basis functions when actually used
- Cache intermediate results

**Implementation**:
```python
class LazyBasisFunction(BasisFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._computed_value = None
        self._last_input_hash = None
    
    def transform(self, X_processed, missing_mask):
        current_hash = self._compute_hash(X_processed, missing_mask)
        if self._last_input_hash != current_hash or self._computed_value is None:
            self._computed_value = super().transform(X_processed, missing_mask)
            self._last_input_hash = current_hash
        return self._computed_value
    
    def _compute_hash(self, X, mask):
        return hash(X.tobytes() + mask.tobytes())
```

## Implementation Priority

### High Priority (v1.1.0)
1. **Basis Function Caching** - Will have immediate impact on repeated evaluations
2. **Vectorized Basis Matrix Construction** - Improves memory usage and reduces fragmentation
3. **NumPy Optimization** - Replaces inefficient Python loops with vectorized operations

### Medium Priority (v1.2.0)
1. **Memory Pool Allocation** - Reduces memory fragmentation for temporary arrays
2. **Lazy Evaluation** - Avoids unnecessary computations
3. **Improved Candidate Generation** - Optimizes knot selection and filtering

### Low Priority (v1.3.0+)
1. **Parallel Processing** - For basis function evaluation across multiple cores
2. **Sparse Matrix Support** - For large datasets with many features
3. **GPU Acceleration** - Using CuPy or similar libraries

## Performance Testing Framework

### Benchmark Suite
- Create comprehensive benchmarks for each optimization
- Track performance improvements over time
- Compare against baseline performance before optimization

### Profiling Tools
- Use cProfile for CPU profiling
- Use memory_profiler for memory usage tracking
- Use line_profiler for line-by-line analysis
- Use pytest-benchmark for regression testing

### Performance Metrics
- Execution time (seconds)
- Memory usage (MB)
- Peak memory allocation
- Number of function calls
- Cache hit rates (for cached implementations)

## Validation Strategy

### Correctness Testing
- Ensure all optimizations maintain numerical accuracy
- Run existing test suite after each optimization
- Compare results with baseline implementation

### Performance Validation
- Measure performance improvements with benchmarks
- Validate memory usage reductions
- Ensure no performance regressions in other areas

### Compatibility Testing
- Verify scikit-learn compatibility maintained
- Test with various data types and sizes
- Ensure API remains unchanged

## Timeline

### Phase 1: Foundation Optimizations (1-2 weeks)
- Implement basis function caching
- Optimize basis matrix construction
- Replace Python loops with NumPy operations

### Phase 2: Memory Optimizations (2-3 weeks)
- Implement memory pooling
- Add lazy evaluation
- Optimize candidate generation

### Phase 3: Advanced Optimizations (3-4 weeks)
- Add parallel processing support
- Implement sparse matrix support
- Explore GPU acceleration possibilities

## Expected Improvements

### Performance Gains
- **20-30%** speed improvement from basis function caching
- **15-25%** memory reduction from vectorized operations
- **10-20%** speed improvement from NumPy optimization
- **5-15%** additional gains from memory pooling and lazy evaluation

### Overall Impact
- **Total expected improvement**: 50-90% faster execution
- **Memory usage reduction**: 30-50% less peak memory
- **Scalability**: Better performance on larger datasets

## Monitoring and Maintenance

### Continuous Performance Tracking
- Integrate performance benchmarks into CI/CD pipeline
- Alert on performance regressions
- Track improvements over time

### Regular Profiling
- Profile regularly to identify new bottlenecks
- Update optimization strategies as needed
- Maintain performance documentation

### Community Feedback
- Gather performance feedback from users
- Prioritize optimizations based on real-world usage
- Document common performance patterns and solutions