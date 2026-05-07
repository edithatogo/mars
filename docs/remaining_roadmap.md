# Remaining Roadmap

The remaining roadmap is organized around the open Conductor tracks. Completed
tracks are archived in Conductor and no longer listed here.

The active priorities are:

- keep the release inventory and publication handoff aligned with the current
  registry state
- finish the remaining R submission path when the maintainer is ready to
  publish externally
- let the Julia General review period complete, then update the release
  inventory and handoff record with the registry outcome
- keep CI and supply-chain checks aligned with the manifests
- preserve the current `pymars` import API while the public package family
  stays under `mars-earth`
- keep the release inventory, publication handoff, and release checklist
  visible from the docs home page
- keep the completed SOTA lanes visible in archive form for later reference

The completed SOTA lanes are:

| Lane | Archived track | Dependency gate that was satisfied |
| --- | --- | --- |
| Citation and paper metadata | [citation_metadata_joss_packet_20260506](../conductor/archive/citation_metadata_joss_packet_20260506/) | current release inventory and canonical package names |
| Supply-chain evidence | [supply_chain_scorecard_sbom_20260506](../conductor/archive/supply_chain_scorecard_sbom_20260506/) | current CI and release workflows |
| HPC packaging feasibility | [hpc_packaging_feasibility_20260506](../conductor/archive/hpc_packaging_feasibility_20260506/) | release metadata and reproducible build commands |
| ABI and Arrow interoperability | [abi_arrow_interop_feasibility_20260506](../conductor/archive/abi_arrow_interop_feasibility_20260506/) | Rust runtime ownership boundary and binding contract |
| Community governance packets | [community_governance_submission_packets_20260506](../conductor/archive/community_governance_submission_packets_20260506/) | citation metadata, governance baseline, and evidence links |
| Workspace automation export | [workspace_automation_export_20260506](../conductor/archive/workspace_automation_export_20260506/) | lane taxonomy and authenticated Linear/Notion CLI access |

The lane dependency graph and execution model are documented in the
[SOTA Dependency and Parallelization Plan](sota_dependency_parallelization_plan.md).

The completed SOTA planning artifacts are:

- [Workspace Automation](workspace_automation.md)
- [SOTA Dependency and Parallelization Plan](sota_dependency_parallelization_plan.md)
- [Community Submission Readiness](community_submission_readiness.md)
- [Release Inventory](release_inventory.md)
- [Publication Handoff](publication_handoff.md)
- [Release Checklist](release_checklist.md)
- [Ecosystem and Foundation Alignment](ecosystem_foundation_alignment.md)
- [Rust Migration and ABI Compatibility](rust_migration_abi_compatibility.md)
- [SOTA HPC Roadmap](hpc_sota_roadmap.md)
