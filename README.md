# pymars Documentation

This branch contains the complete documentation for the pymars library, built with MkDocs and mkdocstrings for automated API documentation from source code docstrings.

## Overview

The pymars documentation provides:

- Complete API reference extracted from Python source code
- Tutorials for basic and advanced usage
- Installation and configuration guidance
- Health economics-specific application examples
- Model interpretation and visualization guides

## Building Documentation Locally

To build the documentation locally:

1. Install documentation dependencies:
   ```bash
   pip install -r docs/requirements.txt
   ```

2. Serve documentation with live reloading:
   ```bash
   mkdocs serve
   ```

3. Build static site:
   ```bash
   mkdocs build
   ```

The documentation will be available at http://localhost:8000 when served.

## Deployment

The documentation is automatically deployed to GitHub Pages when changes are pushed to this `docs` branch. The GitHub Actions workflow `.github/workflows/docs.yml` in the main branch handles the deployment.

## Documentation Structure

- `index.md`: Main landing page
- `installation.md`: Installation instructions
- `usage.md`: General usage指南
- `api.md`: Auto-generated API documentation
- `tutorials/basic.md`: Basic usage tutorial
- `tutorials/advanced.md`: Advanced features tutorial
- `docs/`: Source files for documentation pages

## Contributing to Documentation

To contribute to documentation:

1. Create or modify markdown files in the appropriate directory
2. Test changes locally with `mkdocs serve`
3. Update the navigation as needed in `mkdocs.yml`
4. Submit a pull request to the `docs` branch

## Technology Stack

- **MkDocs**: Static site generator for project documentation
- **Material for MkDocs**: Theme providing responsive, mobile-friendly design
- **mkdocstrings**: Automatic API documentation from Python docstrings
- **Python 3.11**: Required for documentation build process

## Key Features

1. **Automatic API Generation**: Docstrings from pymars source code automatically appear in documentation
2. **Cross-Reference Support**: Automatic linking between related documentation sections
3. **Search Capability**: Full-text search across all documentation
4. **Responsive Design**: Mobile-friendly layout for all devices
5. **Dark/Light Modes**: User preference-based theme selection

## License

The documentation is released under the same MIT License as the pymars library.