# Community Submission Readiness

This page turns the scientific-stewardship roadmap into concrete submission
prep. It does not submit the project anywhere; it defines what must be ready
before a maintainer starts an external process.

## Submission Matrix

| Target | Current fit | Missing before submission |
| --- | --- | --- |
| scikit-learn-contrib | Strong Python estimator fit and published PyPI package | concise contributor guide, compatibility checklist, issue template, and maintainer commitment statement |
| pyOpenSci | Good scientific package candidate if positioned around reproducible mars modeling workflows | pre-submission inquiry, scientific workflow examples, dependency policy summary, and review-ready docs map |
| rOpenSci | R package is locally check-ready and tied to a scientific algorithm | r-universe/CRAN publication decision, package-review summary, R examples, and maintenance policy |
| NumFOCUS | Plausible long-term umbrella if community and governance mature | governance model, code of conduct, steering/maintainer model, fiscal needs, and sustainability statement |
| JOSS | Plausible once external package state is stable and the paper scope is clear | `paper.md`, `paper.bib`, statement of need, research impact, release DOI, and citation metadata |
| speck / Spack | `speck` is tracked as requested; Spack is the concrete HPC packaging target | Spack package recipe feasibility, source tarball policy, compiler/platform notes, and install smoke test |
| EasyBuild | Plausible HPC packaging path after source tarball and dependency policy stabilize | EasyBuild easyconfig feasibility, dependency list, module naming, and HPC install smoke test |

## Required repo Artifacts

| Artifact | Purpose | Status |
| --- | --- | --- |
| `CITATION.cff` | citation metadata for JOSS, Zenodo, and scientific users | recommended |
| `codemeta.json` | machine-readable software metadata | recommended |
| `paper.md` and `paper.bib` | JOSS paper source | recommended before JOSS |
| `docs/community_submission_readiness.md` | community submission checklist | present |
| `docs/release_metadata.json` | canonical package/version state | present |
| `docs/release_inventory.md` | human-readable release state | present |
| `CONTRIBUTING.md` | contributor workflow | present, should be reviewed before submission |
| Code of conduct | community governance expectation | recommended if not already present |

## Community-Specific Notes

scikit-learn-contrib should be treated as the Python-facing compatibility home,
not as the umbrella for every language binding.

pyOpenSci should be approached with a pre-submission inquiry because the package
is a scientific ML implementation with a polyglot runtime story. The inquiry
should explain the Python surface, Rust core, and cross-language conformance
fixtures.

rOpenSci should be approached through the R package surface, after the R
submission path is settled. The review story should emphasize local
`R CMD check`, package manual, vignette, examples, and reproducible fixtures.

NumFOCUS should wait until governance is explicit. A project can be technically
strong and still not ready for a foundation application if governance,
maintainer continuity, and sustainability are not documented.

JOSS should wait until a stable release DOI exists and the statement of need is
sharp. The paper should describe why a maintained, polyglot mars implementation
is useful for scientific modeling.

Spack and EasyBuild should be treated as HPC packaging readiness, not as proof
that the library is an HPC runtime. They make installation in HPC environments
easier; they do not imply GPU, TPU, MPI, or distributed execution support.

## References

- scikit-learn-contrib: https://github.com/scikit-learn-contrib/scikit-learn-contrib
- pyOpenSci package scope: https://www.pyopensci.org/software-peer-review/about/package-scope.html
- rOpenSci software review: https://ropensci.org/software-review/
- NumFOCUS: https://numfocus.org/
- JOSS submission requirements: https://joss.readthedocs.io/en/latest/submitting.html
- JOSS paper format: https://joss.readthedocs.io/en/latest/paper.html
- Spack packaging guide: https://spack.readthedocs.io/en/latest/packaging_guide_creation.html
- EasyBuild docs: https://docs.easybuild.io/
