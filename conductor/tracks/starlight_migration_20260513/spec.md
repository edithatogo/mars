# Spec: Starlight Docs Migration

**Track:** `starlight_migration_20260513`
**Requirement:** REQ-MIG-003
**Status:** Completed 2026-05-13

## Objective

Migrate the `mars` (pymars) documentation site from MkDocs Material to
Starlight+polyglot, enabling:

- Python API documentation generated from source code via polyglot
- Modern Astro-based toolchain with plugin ecosystem
- Sitemap generation for SEO (`@astrojs/sitemap`)
- LLM-friendly documentation export (`starlight-llms-txt`)
- Versioned documentation support (`starlight-versions`)

## Scope

### In Scope

- Create `docs/astro-site/` with full Starlight scaffold
- Convert all 74 MkDocs Markdown files to MDX in `src/content/docs/`
- Configure sidebar matching the existing MkDocs nav structure
- Set up `starlight-polyglot` for Python (`pymars`) API docs
- Update `.github/workflows/docs.yml` to build with pnpm + Astro
- Update `conductor/tech-stack.md` to mark Starlight as ACTIVE
- Update `docs/starlight_docs_stack.md` to note migration completed

### Out of Scope

- Converting MkDocs-specific admonitions/callouts to Starlight format
- Deploying the site (handled by existing GH Pages workflow)
- Content rewrites or editorial changes

## Deliverables

| Artifact | Location |
|---|---|
| Starlight scaffold | `docs/astro-site/` |
| Converted MDX pages | `docs/astro-site/src/content/docs/` (74 pages) |
| Astro config with sidebar | `docs/astro-site/astro.config.mjs` |
| CI/CD workflow | `.github/workflows/docs.yml` (updated) |
| Conductor track | `conductor/tracks/starlight_migration_20260513/` |

## Dependencies

- `@astrojs/starlight ^0.39.0`
- `@astrojs/sitemap ^3.3.0`
- `starlight-polyglot ^0.1.0` (file: link to local package)
- `starlight-links-validator ^0.19.2`
- `starlight-versions ^0.8.0`
- `starlight-llms-txt ^1.0.0`
- Node.js >= 22.12.0
- pnpm >= 10.32.1

## Validation

1. `pnpm install --frozen-lockfile` succeeds
2. `pnpm run build` produces `site/` directory with all pages
3. Sidebar matches original MkDocs nav structure
4. CI workflow builds and deploys Astro site
5. Original `docs/` directory preserved as backup
