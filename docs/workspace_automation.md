# Workspace Automation

This page defines the supported terminal automation path for Linear and Notion.
The repo remains the canonical source for code, Conductor tracks, and durable
documentation. Linear and Notion should track work and decisions without
becoming duplicate source-of-truth systems.

## Installed CLIs

The local workspace has two standalone CLIs available:

| Tool | Binary | Installed by | Primary use |
| --- | --- | --- | --- |
| Notion | `notionctl` and `notion` | `pipx install notionctl` | create and maintain pages, databases, blocks, comments, users, teams, search, and raw Notion API calls |
| Linear | `~/.cargo/bin/linear-cli` | `cargo install linear-cli` | manage issues, projects, documents, roadmaps, initiatives, milestones, labels, comments, and raw Linear GraphQL calls |

Authentication is external to the repo:

```bash
notion auth login
~/.cargo/bin/linear-cli auth
```

`notionctl` can also use `NOTION_API_KEY`. `linear-cli` stores its Linear
credentials through its own auth flow.

## Linear Operating Model

Use Linear for issue and roadmap execution. The recommended structure is:

| Area | Linear object | Purpose |
| --- | --- | --- |
| Core runtime | Project or initiative | Rust core, ABI, profiling, and performance work |
| Bindings | Project | Python, R, Julia, Rust, C#, Go, and TypeScript package work |
| Scientific stewardship | Initiative | scikit-learn-contrib, pyOpenSci, rOpenSci, NumFOCUS, JOSS, speck/Spack, and EasyBuild readiness |
| Ecosystem alignment | Initiative | Apache Arrow, PyPA, .NET Foundation, Julia communities, and R communities |
| Release governance | Project | registry state, submission blockers, and post-publish verification |
| Workspace operations | Project | Notion/Linear maintenance and review passes |

Recommended labels:

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

Ready-for-work issues should include:

- linked Conductor track or doc page
- acceptance criteria
- validation command
- owner or maintainer action if external

Ready-for-review issues should include:

- changed files
- validation output
- remaining blocker if any
- whether external auth or registry action is required

## Notion Operating Model

Use Notion for curated knowledge, decision logs, and submission prep. The
recommended top-level page is `mars Knowledge Base` with these child pages:

| Page | Purpose |
| --- | --- |
| Roadmap Index | links to Conductor, docs, release metadata, and active community plans |
| Scientific Stewardship | submission-readiness notes for scientific communities |
| Ecosystem Alignment | Apache Arrow, PyPA, .NET Foundation, Julia, and R community positioning |
| HPC and ABI | performance, ABI, Spack/EasyBuild, HPSF, and E4S notes |
| Decisions | durable decisions with date, owner, context, and consequence |
| Release State | human-readable mirror of `docs/release_metadata.json` |
| Workspace Reviews | Linear and Notion review notes by phase |

Canonical content stays in the repo. Notion pages should link back to source
docs instead of copying them wholesale.

## Review Cadence

The Linear/Notion workspace track uses four phases:

1. Linear development
2. Linear review
3. Notion development
4. Notion review

Each review should confirm:

- the workspace structure maps back to Conductor tracks
- no page or issue duplicates canonical repo docs
- community targets are visible but not noisy
- external blockers are marked as external, with owner/action/date

## Additional Recommendation

Add generated workspace snapshots later, not now. Once Linear and Notion are
authenticated, a small script can export the active Linear roadmap and Notion
index into `docs/workspace_snapshot.md`. That should be read-only evidence, not
a second planning system.

## References

- Notion CLI: local `notionctl --help`
- Linear CLI: local `linear-cli --help`
