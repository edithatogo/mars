# Plan: migrate_starlight

## Steps

### Step 1: Create directory structure ✅

- [x] `docs/astro-site/` created
- [x] `src/content/docs/`, `src/styles/`, `public/` created

### Step 2: Configure Astro/Starlight ✅

- [x] `package.json` with dependencies
- [x] `astro.config.mjs` with Starlight + polyglot
- [x] `public/.nojekyll` for GitHub Pages

### Step 3: Write documentation content ✅

- [x] `index.mdx` — Overview with feature cards
- [x] `getting-started.mdx` — Installation guide
- [x] `api-reference.mdx` — Complete API docs

### Step 4: Integrate starlight-polyglot ✅

- [x] Go handler configured in `astro.config.mjs`
- [x] Module path points at `../../` (Mars root)
- [x] Local dependency via `file:` protocol

### Step 5: Custom styling ✅

- [x] Red theme accent colors (`#c0392b`, `#e74c3c`)
- [x] Card hover effects
- [x] Mars favicon

### Step 6: Deployment ✅

- [x] `docs.yml` workflow for GitHub Pages
- [x] Build + deploy steps
- [x] `site` and `base` configured

### Step 7: Verify

- [ ] `pnpm install` resolves dependencies
- [ ] `pnpm run build` succeeds
- [ ] Navigation works for all pages
- [ ] Polyglot generates Go API pages
