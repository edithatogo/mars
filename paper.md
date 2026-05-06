---
title: "mars-earth: A Pure Python Implementation of Multivariate Adaptive Regression Splines"
authors:
  - name: Dylan A. Mordaunt
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent
bibliography: paper.bib
date: 2026-05-07
---

# Summary

`mars-earth` is a pure Python implementation of multivariate adaptive regression
splines (MARS). It preserves the `pymars` import namespace for existing code
while packaging the project under the `mars-earth` distribution name. The
library exposes a scikit-learn compatible estimator API and portable model
specification tooling for fitting, serializing, validating, and replaying MARS
models.

# Statement of Need

MARS models are useful when nonlinear structure and interactions matter, but
many implementations depend on compiled extensions or narrow runtime
environments. That dependency profile can make installation and deployment
harder than the modeling task itself. `mars-earth` addresses that problem by
providing a pure Python codebase with a familiar estimator interface, so users
can fit and deploy models without giving up cross-platform portability.

The project also keeps the historical Python import path stable. Existing code
can continue to use `import pymars as earth` while the distribution name remains
`mars-earth`, which helps downstream projects migrate without changing their
application code.

# Implementation

The package centers on the `Earth` estimator and the runtime helpers exposed
through `pymars.runtime`. Model specifications are represented as portable JSON
objects, which allows a fitted model to be serialized, validated, reloaded, and
replayed in a deterministic way across environments.

The implementation follows scikit-learn conventions for estimator parameters,
input validation, and prediction APIs. It also keeps the portable model-spec
format explicit so that downstream tooling can inspect or reuse fitted models
without depending on internal Python objects.

# Related Work

`mars-earth` is inspired by Friedman's original MARS formulation [@Friedman1991]
and by scikit-learn's estimator conventions [@Pedregosa2011].
