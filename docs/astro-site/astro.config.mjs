import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import sitemap from '@astrojs/sitemap';
import starlightLinksValidator from 'starlight-links-validator';
import starlightVersions from 'starlight-versions';
import starlightLlmsTxt from 'starlight-llms-txt';
import polyglot from 'starlight-polyglot';

export default defineConfig({
  site: 'https://edithatogo.github.io/mars',
  base: '/mars',
  outDir: '../../site',
  publicDir: './public',
  srcDir: './src',
  trailingSlash: 'never',
  integrations: [
    sitemap(),
    starlightLinksValidator(),
    starlightLlmsTxt(),
    starlightVersions(),
    starlight({
      title: 'mars Documentation',
      description: 'Multivariate Adaptive Regression Splines — mars (pymars) library documentation',
      logo: {
        replacesTitle: true,
      },
      social: {
        github: 'https://github.com/edithatogo/mars',
      },
      editLink: {
        baseUrl: 'https://github.com/edithatogo/mars/edit/main/docs/',
      },
      plugins: [
        polyglot({
          python: {
            entryPoints: ['pymars'],
          },
        }),
      ],
      sidebar: [
        {
          label: 'Home',
          link: '/',
        },
        {
          label: 'Installation',
          link: '/installation',
        },
        {
          label: 'Usage',
          link: '/usage',
        },
        {
          label: 'Tutorials',
          items: [
            { label: 'Overview', link: '/tutorials/' },
            { label: 'Basic', link: '/tutorials/basic' },
            { label: 'Advanced', link: '/tutorials/advanced' },
          ],
        },
        {
          label: 'API Reference',
          autogenerate: { directory: 'api' },
        },
        {
          label: 'Bindings',
          items: [
            { label: 'Overview', link: '/bindings' },
            { label: 'Release', link: '/binding_release' },
            { label: 'Backend Decisions', link: '/binding_backend_decisions' },
            { label: 'ABI and API Contract', link: '/binding_abi_contract' },
            { label: 'ABI and Arrow Decision', link: '/binding_abi_arrow_decision' },
          ],
        },
        {
          label: 'HPC',
          collapsed: false,
          items: [
            { label: 'SOTA HPC Roadmap', link: '/hpc_sota_roadmap' },
            { label: 'Contracts', link: '/hpc_contracts' },
            { label: 'Parallel Execution Guide', link: '/hpc_parallel_execution_guide' },
            { label: 'CPU Parallel Runtime Benchmarks', link: '/hpc_cpu_parallel_runtime_benchmarks' },
            { label: 'Claim Review Checklist', link: '/hpc_claim_review_checklist' },
            { label: 'Track Checkpoint Notes', link: '/hpc_track_checkpoint_notes' },
            { label: 'HPSF/E4S Readiness Packets', link: '/hpsf_e4s_readiness_packets_20260511' },
          ],
        },
        {
          label: 'Contributor Guides',
          collapsed: false,
          items: [
            { label: 'CI and Quality Policy', link: '/ci_quality' },
            { label: 'Package Release Paths', link: '/package_release_paths' },
            { label: 'Supply Chain Security', link: '/supply_chain' },
            { label: 'Release Checklist', link: '/release_checklist' },
            { label: 'Release Inventory', link: '/release_inventory' },
          ],
        },
        {
          label: 'Architecture',
          collapsed: false,
          items: [
            { label: 'Design', link: '/design' },
            { label: 'Rust Core', link: '/rust_core' },
            { label: 'Rust Core Ownership Boundary', link: '/rust_core_ownership' },
            { label: 'Rust Core Full Conversion Boundary', link: '/rust_core_full_conversion_boundary' },
            { label: 'Rust Core Observability', link: '/rust_core_observability' },
            { label: 'Core Transition Evidence', link: '/core_transition_evidence' },
            { label: 'Training Core Migration', link: '/training_core_migration' },
            { label: 'Training Orchestration Inventory', link: '/training_orchestration_inventory' },
          ],
        },
      ],
    }),
  ],
});