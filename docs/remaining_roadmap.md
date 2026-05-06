# Remaining Roadmap

The remaining roadmap is organized around the open Conductor tracks. Completed
tracks are archived in Conductor and no longer listed here.

The active priorities are:

- complete the remaining release publication handoff for Julia while the
  mandatory General review period runs
- update the release inventory when the Julia General registry result lands
- keep CI and supply-chain checks aligned with the manifests
- preserve the current `pymars` import API while the public package family
  stays under `mars-earth`
- execute the six SOTA follow-on tracks below using independent subagent lanes
  with explicit dependency gates and file ownership

The active SOTA lanes are:

| Lane | Track | Dependency gate |
| --- | --- | --- |
| Citation and paper metadata | `citation_metadata_joss_packet_20260506` | current release inventory and canonical package names |
| Supply-chain evidence | `supply_chain_scorecard_sbom_20260506` | current CI and release workflows |
| HPC packaging feasibility | `hpc_packaging_feasibility_20260506` | release metadata and reproducible build commands |
| ABI and Arrow interoperability | `abi_arrow_interop_feasibility_20260506` | Rust runtime ownership boundary and binding contract |
| Community governance packets | `community_governance_submission_packets_20260506` | citation metadata, governance baseline, and evidence links |
| Workspace automation export | `workspace_automation_export_20260506` | lane taxonomy and authenticated Linear/Notion CLI access |

The lane dependency graph and subagent ownership model are documented in the
[SOTA Dependency and Parallelization Plan](sota_dependency_parallelization_plan.md).

The completed SOTA planning artifacts are:

- [Workspace Automation](workspace_automation.md)
- [SOTA Dependency and Parallelization Plan](sota_dependency_parallelization_plan.md)
- [Community Submission Readiness](community_submission_readiness.md)
- [Ecosystem and Foundation Alignment](ecosystem_foundation_alignment.md)
- [Rust Migration and ABI Compatibility](rust_migration_abi_compatibility.md)
- [SOTA HPC Roadmap](hpc_sota_roadmap.md)
