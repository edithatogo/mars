# Community Submission Readiness

This page turns the scientific-stewardship roadmap into concrete submission
prep. It does not submit the project anywhere; it defines what must be ready
before a maintainer starts an external process.

The lane is aligned with the citation-metadata work, but it does not edit the
citation files themselves. It expects the citation lane to own:

- `CITATION.cff`
- `codemeta.json`
- `paper.md`
- `paper.bib`

This document only explains how those artifacts are used in the submission
packets below.

## Shared Packet Ingredients

Most external review packets need some combination of:

- a short project summary
- a maintenance and support statement
- a contributor and governance model
- release and packaging evidence
- a citation and authorship pack
- reproducible examples or smoke tests
- a clear statement of what the Rust core owns and what Python keeps as
  adapter glue

The packet material should agree with `docs/release_metadata.json`,
`docs/release_inventory.md`, `docs/package_release_paths.md`,
`docs/hpc_contracts.md`, and `docs/community_submission_readiness.md`.

## Submission and Alignment Matrix

| Target | Packet type | Current fit | Missing or unresolved gate |
| --- | --- | --- | --- |
| scikit-learn-contrib | Submission packet | Strong Python estimator fit and a published PyPI package | concise contributor guide, compatibility checklist, issue template, and maintainer commitment statement |
| pyOpenSci | Pre-submission inquiry packet | Good scientific package candidate if positioned around reproducible mars modeling workflows | pre-submission inquiry, scientific workflow examples, dependency policy summary, and review-ready docs map |
| rOpenSci | Review packet | R package is locally check-ready and tied to a scientific algorithm | final r-universe/CRAN publication path, package-review summary, R examples, and maintenance policy |
| NumFOCUS | Stewardship packet | Plausible long-term umbrella if governance and continuity are mature | governance model, code of conduct, steering/maintainer model, fiscal needs, and sustainability statement |
| JOSS | Paper packet | Plausible once package state and paper scope are stable | `paper.md`, `paper.bib`, statement of need, research impact, release DOI, and citation metadata |
| PyPA | Packaging-alignment packet | Python packaging baseline is already in place | keep `pyproject.toml`, trusted publishing, wheel smoke tests, and metadata aligned with PyPA specs |
| .NET Foundation | Stewardship packet | C# binding exists and is published | documented C# ownership/support policy, project-health summary, and maintainer statement |
| Julia | Registry and community packet | Julia package registration is pending in Julia General (target identity `MarsEarth`, with `MarsRuntime` kept as superseded legacy) | keep package metadata, examples, issue routing, release notes, and registration status current for future releases |
| R | Package-review packet | R package is locally publication-ready | keep `R CMD check`, manual, vignette, examples, and maintainer notes current for r-universe/CRAN review |
| HPSF | HPC readiness packet | Benchmarking and Rust-core observability now exist | benchmark artifacts, portability notes, ABI story, and install smoke tests for HPC consumers |
| E4S | HPC packaging readiness packet | Feasible only if packaging and portability evidence stay stable | packaging recipes, install smoke tests, dependency policy, and reproducible build notes |

## Community-Specific Notes

scikit-learn-contrib should be treated as the Python-facing compatibility home,
not as the umbrella for every language binding.

pyOpenSci should be approached with a pre-submission inquiry because the
package is a scientific ML implementation with a polyglot runtime story. The
inquiry should explain the Python surface, Rust core, and cross-language
conformance fixtures.

rOpenSci should be approached through the R package surface after the R
submission path is settled. The review story should emphasize local `R CMD
check`, the package manual, vignette coverage, examples, and reproducible
fixtures.

NumFOCUS should wait until governance is explicit. A project can be technically
strong and still not be ready for a foundation application if governance,
maintainer continuity, and sustainability are not documented.

JOSS should wait until a stable release DOI exists and the statement of need is
sharp. The paper should describe why a maintained, polyglot mars implementation
is useful for scientific modeling.

PyPA should be treated as packaging and interoperability alignment, not as a
submission venue. The target is to keep wheels, metadata, and trusted
publishing aligned with PyPA specs.

.NET Foundation should be treated as ecosystem stewardship and project-health
alignment, not as a package registry gate.

Julia should be treated as a pending registry registration through Julia
General (`MarsEarth` target, `MarsRuntime` retained as superseded legacy). Keep
the package metadata, examples, and maintainer notes current so future releases
are ready.

R should be treated as a package-review and publication workflow through
r-universe and CRAN.

HPSF and E4S should be treated as HPC packaging and portability readiness
packets until higher contract levels in [HPC Contracts](hpc_contracts.md) are
implemented. They make installation easier for HPC users; this remains H0-only and
does not imply H0-level GPU, TPU, MPI, or distributed execution support.

## Required repository artifacts

| Artifact | Purpose | Status |
| --- | --- | --- |
| `CITATION.cff` | citation metadata for JOSS, Zenodo, and scientific users | expected from the citation lane |
| `codemeta.json` | machine-readable software metadata | expected from the citation lane |
| `paper.md` and `paper.bib` | JOSS paper source | expected before JOSS |
| `docs/community_submission_readiness.md` | community submission checklist | present |
| `docs/release_metadata.json` | canonical package/version state | present |
| `docs/release_inventory.md` | human-readable release state | present |
| `CONTRIBUTING.md` | contributor workflow | present and should be reviewed before submission |
| `CODE_OF_CONDUCT.md` | community conduct expectation | present |
| `GOVERNANCE.md` | maintainer and decision model | present or should be added by the community lane |
| `SUPPORT.md` | issue and maintainer support routing | present or should be added by the community lane |

## External Review Gates

These items cannot be completed purely from source edits:

- scikit-learn-contrib: maintainer review and community fit decision
- pyOpenSci: pre-submission response and peer review
- rOpenSci: review queue decision and maintainer follow-through
- NumFOCUS: stewardship and governance application review
- JOSS: editorial and reviewer decision
- Julia: General registry publication
- R: r-universe and CRAN submission/review decision
- HPSF and E4S: external packaging community review and feedback

## References

- scikit-learn-contrib: https://github.com/scikit-learn-contrib/scikit-learn-contrib
- pyOpenSci package scope: https://www.pyopensci.org/software-peer-review/about/package-scope.html
- rOpenSci software review: https://ropensci.org/software-review/
- NumFOCUS: https://numfocus.org/
- JOSS submission requirements: https://joss.readthedocs.io/en/latest/submitting.html
- JOSS paper format: https://joss.readthedocs.io/en/latest/paper.html
- PyPA specifications: https://packaging.python.org/specifications/
- .NET Foundation membership policy: https://dotnetfoundation.org/about/policies/.net-foundation-membership-policy
- Julia package registration: https://help.juliahub.com/juliahub/stable/registering/
- r-universe documentation: https://docs.r-universe.dev/
- Spack packaging guide: https://spack.readthedocs.io/en/latest/packaging_guide_creation.html
- EasyBuild docs: https://docs.easybuild.io/
