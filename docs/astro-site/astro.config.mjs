import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';
import polyglot from 'starlight-polyglot';

export default defineConfig({
  site: 'https://edithatogo.github.io/mars',
  base: '/mars',
  integrations: [
    starlight({
      title: 'Mars',
      description: 'Data analysis toolkit for structured time-series data',
      favicon: '/favicon.svg',
      logo: {
        source: 'local',
        src: '/public/favicon.svg',
      },
      social: {
        github: 'https://github.com/edithatogo/mars',
      },
      plugins: [
        starlightLinksValidator(),
      ],
      sidebar: [
        {
          label: 'Getting Started',
          items: [
            { label: 'Overview', slug: 'index' },
            { label: 'Getting Started', slug: 'getting-started' },
          ],
        },
        {
          label: 'API Reference',
          items: [
            { label: 'Overview', slug: 'api-reference' },
          ],
        },
      ],
      customCss: [
        './src/styles/custom.css',
      ],
    }),
    polyglot({
      handlers: [
        {
          name: 'go',
          handler: 'go',
          options: {
            modulePath: '../../',
            output: './src/content/docs/',
            pagination: true,
          },
        },
      ],
    }),
  ],
});
