# Starlight Docs Stack Governance

This page records the current evaluation state for a possible Starlight-based
documentation stack. It is a governance document, not a migration commitment.

The current live docs site remains mkdocs Material. Any Starlight adoption must
be approved explicitly and version-pinned before it replaces or coexists with
the existing site.

## Core Version Target

- `@astrojs/starlight` `0.38.5`

This is the latest Starlight release we have verified against the public
release notes and package metadata. If the project adopts Starlight later, the
exact version should still be checked again before a migration lands.

## Required and Relevant Plugins

### Required for governance

- `starlight-links-validator` `0.19.2`

Use this to keep internal links and docs references checked in CI.

### Required if versioned docs are approved

- `starlight-versions` `0.8.0`

Use this only if the docs site needs versioned content. The plugin currently
requires `Starlight >= 0.38.0`, so the core version target above is compatible.

### Optional depending on future scope

- `@astrojs/starlight-docsearch` `0.6.0`
  - Use if the docs search experience is moved to Algolia DocSearch.
- `starlight-typedoc` `0.21.3`
  - Use if the docs site later publishes TypeScript API reference pages.
- `@astrojs/starlight-tailwind` `4.0.1`
  - Use if custom styling or a Tailwind-based theme layer is needed.

## Validation Policy

If a Starlight migration is approved later, the docs stack should be pinned and
validated with:

- a reproducible install step for the Node/astro toolchain
- a production build command for the Starlight site
- a link-validation step
- a content/version smoke test for any versioned-docs setup

The current mkdocs build remains the source of truth until a migration track
explicitly replaces it.

## Migration Policy

- Keep mkdocs Material live until a Starlight migration is approved.
- Do not treat Starlight as the live docs stack just because the version
  decision is recorded.
- Re-evaluate the Starlight core and plugin versions before a migration is
  implemented.
- Keep versioned docs and search plugins optional until there is a confirmed
  content need.
