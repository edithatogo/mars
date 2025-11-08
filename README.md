# pymars Documentation

This branch contains the documentation for the pymars library, built using MkDocs with mkdocstrings.

## Overview

This branch contains the complete documentation for pymars, including:

- API reference extracted from Python docstrings
- Usage tutorials for both basic and advanced features
- Installation and setup instructions
- Health economics-specific applications

## Building the Documentation

To build the documentation locally:

```bash
# Install dependencies
pip install -r docs/requirements.txt

# Serve documentation with live reloading
mkdocs serve

# Build static site
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

## Contributing to Documentation

Documentation improvements are welcome! To contribute:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Test locally with `mkdocs serve`
5. Submit a pull request

## Structure

- `index.md` - Main landing page
- `docs/` - All documentation content files
  - `tutorials/` - Step-by-step tutorial guides
  - `api.md` - Auto-generated API reference
  - Other documentation pages
- `mkdocs.yml` - Configuration file
- `requirements.txt` - Documentation dependencies

## License

The documentation is licensed under the same license as the main pymars library (MIT License).