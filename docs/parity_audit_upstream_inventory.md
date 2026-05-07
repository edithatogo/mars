# Parity Audit Upstream Inventory

This note records the upstream sources reviewed for the mars / earth parity
audit and the behavior expectations each source contributes. It is a source
inventory, not a findings list.

## Reviewed Sources

| Source | Canonical URL(s) | High-level expectation contributed |
| --- | --- | --- |
| `py-earth` README | https://github.com/scikit-learn-contrib/py-earth | Establishes the upstream project frame: a scikit-learn-style mars implementation, `Earth` as the entry point, missing-data support, and a documented list of known gaps or future improvements. |
| `py-earth` docs landing / API introduction | https://contrib.scikit-learn.org/py-earth/ <br> https://contrib.scikit-learn.org/py-earth/content.html | Defines the documented estimator surface: `Earth` parameters, two-stage forward/pruning training, dense-array and pandas/patsy input handling, scoring, feature-importance summaries, basis transforms, and derivative predictions. |
| R `earth` reference manual | https://cran.r-universe.dev/earth/doc/manual.html | Defines the R package behavior baseline: formula and default interfaces, `summary.earth`, `plot.earth`, `predict.earth`, `evimp`, `update.earth`, variance-model support, and GLM-adjacent extensions. |
| R `earth` package docs landing page | https://search.r-project.org/CRAN/refmans/earth/html/00Index.html <br> https://stephenmilborrow.r-universe.dev/earth | Confirms the public help-page surface and package organization: the core `earth` entry points, plotting and summary methods, update workflows, and the documented package dependency / vignette ecosystem. |

## Inventory Notes

- The `py-earth` sources are the baseline for scikit-learn-compatible estimator
  behavior.
- The R `earth` sources are the baseline for formula-oriented modeling,
  downstream diagnostic methods, and prediction-interval support.
- This inventory stays intentionally high level so later parity documents can
  map specific behaviors back to these upstream references without restating
  the source scan.
