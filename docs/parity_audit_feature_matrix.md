# py-earth Feature Matrix

This note inventories the upstream `py-earth` feature surface for the parity
audit. It stays intentionally narrow: only the upstream `py-earth` README and
documentation are used here, so the matrix can serve as a source-backed
baseline for later gap comparison.

## Scope

- Upstream reference family: `py-earth`
- Sources used:
  - [`py-earth` GitHub README](https://github.com/scikit-learn-contrib/py-earth)
  - [`py-earth` docs introduction/API page](https://contrib.scikit-learn.org/py-earth/content.html)
  - [`py-earth` docs landing page](https://contrib.scikit-learn.org/py-earth/)

## Matrix

| Area | Upstream py-earth capability | Evidence | Audit note |
| --- | --- | --- | --- |
| Core model family | `py-earth` is a Python/Cython implementation of Friedman’s mars algorithm in the style of scikit-learn. The docs describe it as a flexible regression method that searches for interactions and non-linear relationships using a multivariate truncated power spline basis. | README: [lines 281-281](https://github.com/scikit-learn-contrib/py-earth#L281-L281). Docs: [lines 19-20](https://contrib.scikit-learn.org/py-earth/content.html#L19-L20), [22-34](https://contrib.scikit-learn.org/py-earth/content.html#L22-L34), [81-87](https://contrib.scikit-learn.org/py-earth/content.html#L81-L87). | This is the baseline behavior to preserve: scikit-learn-style estimator surface with mars basis expansion and interaction search. |
| Basis-term support | The documented basis terms are constant terms, linear functions of input variables, and hinge functions of input variables. The model terms are described as products of constant, linear, and hinge functions. | Docs: [lines 27-34](https://contrib.scikit-learn.org/py-earth/content.html#L27-L34), [172-173](https://contrib.scikit-learn.org/py-earth/content.html#L172-L173). | This is the upstream basis vocabulary used throughout the docs and examples. |
| Training / pruning / selection | Training is explicitly two-stage: a forward pass searches for terms that locally minimize squared error, then a pruning pass chooses a subset with locally minimal `gcv`. The forward pass and pruning pass each expose record objects. | Docs: [lines 33-34](https://contrib.scikit-learn.org/py-earth/content.html#L33-L34), [85-86](https://contrib.scikit-learn.org/py-earth/content.html#L85-L86), [178-179](https://contrib.scikit-learn.org/py-earth/content.html#L178-L179), [364-369](https://contrib.scikit-learn.org/py-earth/content.html#L364-L369). | The audit should treat forward-pass and pruning behavior as parity-critical because they define the learned model, not just its output format. |
| Training controls | The public `Earth` API exposes `max_terms`, `max_degree`, `allow_missing`, `penalty`, `endspan_alpha`, `endspan`, `minspan_alpha`, `minspan`, `thresh`, `zero_tol`, `min_search_points`, `check_every`, `allow_linear`, `use_fast`, `fast_K`, `fast_h`, `smooth`, `enable_pruning`, `feature_importance_type`, and `verbose`. | Docs: [lines 81-87](https://contrib.scikit-learn.org/py-earth/content.html#L81-L87), [91-163](https://contrib.scikit-learn.org/py-earth/content.html#L91-L163). | These parameters define the documented option surface and should be compared directly against the current repo’s defaults and validation behavior. |
| missingness handling | missingness is a first-class documented feature via `allow_missing=True`. The docs also describe a `missing` argument accepted by fit/predict/score-style methods, with missingness inferred from a pandas DataFrame when `allow_missing` is enabled. | Docs: [lines 100-101](https://contrib.scikit-learn.org/py-earth/content.html#L100-L101), [300-313](https://contrib.scikit-learn.org/py-earth/content.html#L300-L313), [326-337](https://contrib.scikit-learn.org/py-earth/content.html#L326-L337), [447-448](https://contrib.scikit-learn.org/py-earth/content.html#L447-L448). | missingness support is explicitly documented and should be audited as a supported user-facing behavior, not an edge-case implementation detail. |
| Categorical handling | The docs cite Friedman’s technical report on mixed ordinal and categorical variables, and the README lists “better support for categorical predictors” as a requested future improvement. The documented public API does not expose a categorical-specific parameter. | README: [lines 287-294](https://github.com/scikit-learn-contrib/py-earth#L287-L294), [331-341](https://github.com/scikit-learn-contrib/py-earth#L331-L341). Docs: [line 81](https://contrib.scikit-learn.org/py-earth/content.html#L81), [line 73](https://contrib.scikit-learn.org/py-earth/content.html#L73). | The safest reading is that categorical behavior is acknowledged in the lineage, but the documented API surface does not present a dedicated categorical mode. Treat this as a likely parity gap unless later source inspection proves otherwise. |
| Diagnostics and summaries | The documented diagnostics include `summary`, `summary_feature_importances`, `trace`, `pruning_trace`, `score`, `score_samples`, `predict_deriv`, and `transform`. Feature-importance criteria include `gcv`, `rss`, and `nb_subsets`. | Docs: [lines 160-163](https://contrib.scikit-learn.org/py-earth/content.html#L160-L163), [172-180](https://contrib.scikit-learn.org/py-earth/content.html#L172-L180), [423-453](https://contrib.scikit-learn.org/py-earth/content.html#L423-L453). | Diagnostics are part of the documented public surface and should be audited as part of user-visible parity, not as internal-only helpers. |
| Formula / interface ergonomics | The documented public API is estimator-centric rather than formula-centric. The docs emphasize scikit-learn compatibility, array-like inputs, pandas support, and patsy-style compatibility where relevant, but do not present a dedicated formula interface as the primary surface. | Docs: [lines 19-20](https://contrib.scikit-learn.org/py-earth/content.html#L19-L20), [87-87](https://contrib.scikit-learn.org/py-earth/content.html#L87-L87). | The upstream ergonomics are centered on the estimator API, which should remain the baseline for parity comparisons. A formula-based interface would be an extension, not a documented expectation. |
| Package semantics | `py-earth` is described as scikit-learn-compatible, accepts numpy/pandas/patsy inputs, supports pickling, and is documented with Sphinx. The README also describes a source install flow via `setup.py`, and the GitHub repository is archived/read-only. | Docs: [lines 19-20](https://contrib.scikit-learn.org/py-earth/content.html#L19-L20), [87-87](https://contrib.scikit-learn.org/py-earth/content.html#L87-L87). README: [lines 150-152](https://github.com/scikit-learn-contrib/py-earth#L150-L152), [297-301](https://github.com/scikit-learn-contrib/py-earth#L297-L301). Docs landing page: [lines 31-33](https://contrib.scikit-learn.org/py-earth/#L31-L33). | The package semantics to compare against are old-style source distribution + docs + pickle support, not a modern multi-registry release flow. |

## Notes

- The upstream docs consistently frame `py-earth` as a regression method with
  a scikit-learn estimator interface.
- The README’s feature request list is useful audit evidence because it captures
  the upstream maintainer’s own stated gaps.
- The diagnostics surface and the package semantics are important for the
  audit because they affect user expectations even when the underlying math is
  unchanged.

## Use in later phases

Later parity-audit phases should use this matrix as the baseline for:

- feature-by-feature comparison against the current repo
- gap classification into parity-critical, nice-to-have, and intentional
  boundaries
- recommendation synthesis for the Rust-first roadmap
