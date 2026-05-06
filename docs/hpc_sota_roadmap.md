# SOTA HPC Roadmap

The project is currently a portable scientific ML library with a Rust core. It
is not yet an HPC runtime. This roadmap defines what would have to change before
the project could credibly pursue HPSF, E4S, Spack, EasyBuild, or accelerator
positioning.

## Current HPC State

| Area | Current state | Gap |
| --- | --- | --- |
| CPU benchmarks | criterion benchmarks exist for Rust runtime paths | no public trend dashboard or release baseline policy |
| CPU profiling | tracing and profiling guidance exist | no standard local pprof/flamegraph recipe in docs |
| Memory profiling | not first-class | no allocation benchmark or memory regression policy |
| Thread parallelism | not exposed as a public feature | no Rayon or scheduler decision recorded |
| GPU | unsupported | no GPU kernels or portability layer |
| TPU | unsupported | no TPU execution model |
| MPI/distributed | unsupported | no multi-node training or inference model |
| HPC packaging | not packaged for Spack/EasyBuild yet | no package recipe or easyconfig feasibility note |

## Near-Term Work

These steps improve engineering maturity without changing the API:

- publish benchmark baselines as CI artifacts
- add a local flamegraph/pprof recipe for Rust hot paths
- add memory profiling guidance for allocation-heavy paths
- document CPU parallelism options and decide whether Rayon is appropriate
- add Spack and EasyBuild feasibility notes
- add OpenSSF Scorecard and SBOM release evidence

## Mid-Term Work

These steps prepare for HPC packaging and ABI use:

- create a narrow runtime ABI proof-of-concept
- test one non-Python binding against the ABI path
- define Arrow C Data / C Stream feasibility for table inputs
- create a Spack package recipe proof-of-concept
- create an EasyBuild easyconfig proof-of-concept
- add clean-container install smoke tests for Linux HPC-style environments

## Long-Term Work

These steps should only proceed if performance data proves they are worthwhile:

- parallelize Rust training/evaluation kernels where deterministic results can
  be preserved
- evaluate SIMD for basis evaluation
- evaluate GPU kernels only after the CPU kernel profile shows enough work to
  amortize transfer and launch costs
- evaluate MPI/distributed execution only if large multi-node workloads are
  real user requirements
- do not pursue TPU support unless a concrete scientific workflow requires it

## HPSF Readiness

HPSF is relevant only after the project can show a credible high-performance
software story:

- benchmark trend evidence
- profiling and tuning workflow
- reproducible builds
- HPC packaging plan
- governance and maintainer model
- clear relationship to existing HPC projects

## E4S Readiness

E4S is relevant only after the project can show scientific software stack fit:

- Spack packaging feasibility
- CPU/GPU portability story
- reproducible release artifacts
- container or module-friendly install path
- interoperability narrative with the rest of the scientific stack

## ABI and Arrow Position

The ABI and Arrow work should be staged:

1. keep public APIs stable
2. add a narrow ABI for stable runtime primitives
3. validate host-language ownership and error rules
4. evaluate Arrow as optional data interchange
5. defer accelerator backends until profiling justifies them

## References

- HPSF: https://hpsf.io/about/
- E4S: https://e4s.io/
- Spack: https://spack.io/
- Spack packaging guide: https://spack.readthedocs.io/en/latest/packaging_guide_creation.html
- EasyBuild: https://easybuild.io/
- EasyBuild docs: https://docs.easybuild.io/
- Apache Arrow: https://arrow.apache.org/
