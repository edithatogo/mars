# Starlight Docs Stack Governance

> **Migration completed 2026-05-13.** This page now records the governance
> state of the adopted Starlight documentation stack.

The docs site has been migrated from mkdocs Material to Starlight+polyglot.
The migration was executed under track `starlight_migration_20260513`.

## Adopted Stack

- **`@astrojs/starlight` `^0.39.0`** — Core documentation framework
- **`@astrojs/sitemap` `^3.3.0`** — Sitemap generation for SEO
- **`starlight-polyglot` `^0.1.0`** — Plugin for generating API docs from Python
  (`pymars`) source code
- **`starlight-links-validator` `^0.19.2`** — Internal link validation in CI
- **`starlight-versions` `^0.8.0`** — Versioned docs support for future releases
- **`starlight-llms-txt` `^1.0.0`** — LLM-friendly documentation export

## Migration Artifacts

- Starlight site scaffold: `docs/astro-site/`
- Converted content: `docs/astro-site/src/content/docs/` (74 MDX pages)
- Sidebar configured in `docs/astro-site/astro.config.mjs`
- CI/CD: `.github/workflows/docs.yml` builds with pnpm + Node.js

## Validation Requirements

The Starlight build is validated in CI via:

1. `pnpm install --frozen-lockfile` — reproducible dependency install
2. `pnpm run build` — production Starlight build (outputs to `site/`)
3. `starlight-links-validator` — checks internal links during build
4. GitHub Pages deployment via `actions/deploy-pages@v4`

## Migration Policy (Historical)

- Previous stack: mkdocs Material (retired 2026-05-13)
- Backup of previous documentation remains in `docs/` for reference
- All future doc content additions should target `docs/astro-site/src/content/docs/`
