# Workspace Automation

This page defines the supported terminal automation path for Linear and
Notion. The repo remains the canonical source for code, Conductor tracks, and
durable documentation. Linear and Notion should mirror that state, not replace
it.

## Scope Boundaries

- Do not commit tokens, API keys, or workspace secrets.
- Do not treat Linear or Notion as an alternate source of truth for code or
  roadmap status.
- Do not edit shared registry pages from this lane; keep that work in the
  lane-specific export templates and local CLI guidance.

## Workspace Taxonomy

The workspace mirrors the six parallel work lanes already used in the SOTA
dependency plan. The concrete objects are intentionally simple so the export
path can be scripted later without inventing extra structure.

| Conductor lane | Linear home | Notion home | Export intent |
| --- | --- | --- | --- |
| Citation metadata and paper packet | Project or initiative | `Roadmap Index` and `Decisions` | publication packet and authoring status |
| Supply chain scorecard and SBOM | Project | `Release State` | CI, provenance, and evidence snapshots |
| HPC packaging feasibility | Project or initiative | `HPC and ABI` | packaging notes and feasibility evidence |
| ABI and Arrow interop feasibility | Project | `HPC and ABI` | contract notes and proof-of-concept evidence |
| Active HPC contracts and implementation | Project or initiative | `HPC and ABI` | H0-H4 contract status, dependencies, and parallel worker ownership |
| Community submission packets | Initiative | `Scientific Stewardship` and `Ecosystem Alignment` | review packets and maintainer actions |
| Workspace automation export | Project | `Workspace Reviews` | review notes and exported snapshots |

Recommended Linear labels for this lane:

- `area:rust-core`
- `area:bindings`
- `area:docs`
- `area:release`
- `area:community`
- `area:hpc`
- `target:scikit-learn-contrib`
- `target:pyopensci`
- `target:ropensci`
- `target:numfocus`
- `target:joss`
- `target:spack`
- `target:easybuild`
- `target:arrow`
- `target:pypa`
- `target:dotnet-foundation`
- `target:julia`
- `target:r`

## Tooling And Authentication

The local environment already exposes `notionctl` and `notion`. Linear CLI is
expected to be installed separately when the workspace export lane is active.
The repo does not vendor either tool.

| Tool | Install path | Authentication | Notes |
| --- | --- | --- | --- |
| Notion | `pipx install notionctl` or `uv tool install notionctl` | `notion auth login`, `NOTION_API_KEY`, or `--token` | `notionctl` and `notion` both expose the same CLI surface. |
| Linear | `cargo install linear-cli` | `linear-cli auth login`, browser-based authorization, or `LINEAR_API_KEY` | Use `linear-cli auth status` before exporting. If your installed build differs, check `linear-cli --help` first. |

## Linear CLI Setup

The Linear CLI should be treated as a local operator tool. Keep the auth
session external to the repo and use environment variables or the CLI keyring
for reuse.

Typical setup:

```bash
cargo install linear-cli
linear-cli auth login
linear-cli auth status
```

If the installed build supports browser-based authorization / PKCE, prefer the
browser flow for interactive use and use an API key only for scripting:

```bash
export LINEAR_API_KEY="lin_api_..."
```

Workspace export guidance:

```bash
mkdir -p exports/linear
linear-cli export csv -t <TEAM_KEY> -f exports/linear/issues.csv
linear-cli export projects-csv -f exports/linear/projects.csv
```

Keep the export scope narrow:

- issues for active lanes
- projects for lane ownership
- milestones for review gates
- labels for taxonomy

## Notion CLI Setup

The Notion CLI is already available locally. Use it for authenticated page and
database access, and keep the token out of the repository.

Typical setup:

```bash
notion auth login
notion auth status
export NOTION_API_KEY="secret_..."
```

Workspace export guidance:

```bash
mkdir -p exports/notion
notion search "mars" > exports/notion/search.json
notion db get <DATABASE_ID> > exports/notion/database.json
notion db query <DATABASE_ID> > exports/notion/roadmap.json
notion page get <PAGE_ID> > exports/notion/page.json
notion block get <PAGE_ID> > exports/notion/blocks.json
```

Use these commands to capture:

- roadmap indices
- release state mirrors
- decisions and review notes
- evidence pages and handoff summaries

## Export Templates

- `docs/templates/linear_workspace_export.md`
- `docs/templates/notion_workspace_export.md`

These templates are source-controlled skeletons only. They are safe to share
because they do not contain secrets, workspace IDs, or tokens.

## Review Cadence

The Linear / Notion workspace track uses four phases:

1. Linear development
2. Linear review
3. Notion development
4. Notion review

Each review should confirm:

- the workspace structure maps back to Conductor tracks
- no page or issue duplicates canonical repo docs
- community targets are visible but not noisy
- external blockers are marked as external, with owner, action, and date

## Status Flow

Conductor remains the source of truth. Linear and Notion are mirrors that
surface the current state to external collaborators.

| Conductor state | Linear mapping | Notion mapping |
| --- | --- | --- |
| `new` | backlog issue or project intake | Roadmap Index note |
| `in_progress` | active issue or project status | Workspace Reviews entry |
| `blocked` | blocked label or project status | Decisions entry with owner and blocker |
| `done` | completed issue or archived project | Release State mirror or review summary |

Use the Notion `Workspace Reviews` page for phase-by-phase notes and keep the
Linear issue/project status aligned with the current Conductor checkpoint.

## Validation

The workspace lane should validate with:

```bash
uv run mkdocs build --strict
notionctl --help
linear-cli --help
```

If `linear-cli` is not installed in the current environment, document that as a
local setup dependency rather than committing credentials.

## Related Roadmap

- [SOTA Dependency and Parallelization Plan](sota_dependency_parallelization_plan.md)
- [Remaining Roadmap](remaining_roadmap.md)
- [Community Submission Readiness](community_submission_readiness.md)
