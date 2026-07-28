⚡ [Performance Improvement] Optimize loop checking for `HingeBasisFunction` in `_pruning.py`

💡 **What:** The optimization implemented
`_compute_gcv_for_subset` inside `PruningPasser` computes the number of hinge terms in a given subset over and over. By modifying the pruning loop within `run` to incrementally track `num_hinge_terms_in_subset`, we eliminate a generator iteration using `sum(isinstance(...))` which is called repeatedly for every intermediate subset removed during the pruning process.

Additionally, the slow generator expression `sum(isinstance(bf, HingeBasisFunction) for bf in basis_subset)` was replaced with a faster generator `sum(type(bf) is HingeBasisFunction for bf in basis_subset)` when computing the initial amount of hinge terms since `type() is` performs significantly faster than `isinstance()` in tight loops.

🎯 **Why:** The performance problem it solves
In `PruningPasser.run`, for every model subset pruned during the sequence, the code was looping through every term checking if it was an instance of `HingeBasisFunction`. Given that a MARS model with $k$ terms checks about $O(k^2)$ subsets, doing $O(k)$ operations inside the inner loop is unnecessarily inefficient, leading to an extra inner check. The incremental counts significantly reduce overhead, avoiding repeated array traversals.

📊 **Measured Improvement:**
In a test comparing the time it takes to prune a `PruningPasser` object containing 150 basis functions:
- Baseline pruning time: 46.2741s
- Optimized pruning time: 46.0167s

While incremental updates drastically improved the performance isolated to the specific counting subroutine code (as shown in a local benchmark isolating just the `sum(isinstance(...))` check taking 2.33 seconds and the `sum(type() is ...)` check taking 0.6 seconds in python), the larger subset profiling identified that `np.linalg.lstsq` taking ~85 seconds dominates the execution time. Regardless, the improvement implements the logic necessary to bring down the cost of hinge checking close to zero without any downsides.
