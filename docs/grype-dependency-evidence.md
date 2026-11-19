# Grype dependency evidence

Grype supplements the existing audit, test and release gates. The shared action
is pinned to an immutable commit, refreshes its vulnerability database and
retains scanner, database, inventory, source and policy evidence. Pull-request
findings are advisory; candidate publication uses the Critical threshold.
Missing inventories and operational errors fail either mode.

## Python distribution

The release workflow builds the wheel and sdist with Cargo's locked resolution.
`scripts/generate_release_sbom.sh` installs the hash-locked Python runtime graph
with the `cli` and `pandas` extras into a fresh Python 3.12 environment, then
installs the exact candidate wheel without resolving again. A validated
CycloneDX document describes that environment. A separate SPDX document records
the resolved `rust-runtime/Cargo.lock` source graph. Neither substitutes for an
inventory of native system libraries inside every platform wheel.

Both scans must succeed before TestPyPI or PyPI publication. Candidate hashes
are checked before the existing publication steps. The GitHub release retains
`python.sbom.cdx.json`, `rust.sbom.spdx.json` and subject hashes as separate
assets, so JSON evidence is not accidentally sent to PyPI.

## Binding publication

The manual Rust route packages with `--locked`, scans its Cargo graph and
checks the package/lock hashes before locked publication. The TypeScript route
builds one npm tarball, inventories its extracted contents, checks its hash and
publishes that same tarball. The current npm package declares no runtime
dependencies; the inventory does not include a separately installed Rust CLI.
The C# route inventories its built assemblies and `.deps.json` metadata, retains
the internal project identity and binds the exact NuGet package hashes. Its
consumer-provided .NET and Rust runtimes remain outside that package inventory.

The R, Julia and Go binding jobs retain their existing manual publication notes.
Their source declarations are not treated as resolved inventories. The separate
Conda/Anaconda routes are not qualified by this rollout.

## Monitoring and acceptance

The weekly workflow independently scans the two retained SBOM assets from the
latest published non-prerelease release, binding each asset ID and
GitHub-provided SHA-256. Missing assets fail explicitly until a qualified release
has supplied them. Candidate checks cannot stand in for released-asset monitoring.
Caller-owned release input receipts are retained for 30 days and pull-request
input receipts for 14 days; the shared scanner retains its outputs for 90 days. Always-retained partial evidence distinguishes
scanner failures from completed policy results.

Local workflow lint and scoped shell checks do not establish native build,
registry or hosted scanner acceptance. No release is authorized by adding these
workflow definitions; normal release conditions and credentials remain required.
