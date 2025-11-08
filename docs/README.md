# Documentation System Setup

## Overview

The mars repository now has a complete documentation system using MkDocs with mkdocstrings. This provides:

1. **Easy-to-read documentation** in Markdown format
2. **Auto-generated API documentation** from Python docstrings
3. **Tutorials and guides** for users at different levels
4. **Integration with GitHub Pages** for online hosting

## Directory Structure

```
docs/                    # MkDocs documentation source
├── index.md             # Main landing page
├── installation.md      # Installation instructions
├── usage.md             # Usage guide
├── api.md               # Auto-generated API reference
├── tutorials/           # Tutorial files
│   ├── basic.md         # Basic usage tutorial
│   └── advanced.md      # Advanced features tutorial
└── requirements.txt     # Documentation dependencies
```

## Local Development

To build and serve the documentation locally:

```bash
# Install documentation dependencies
pip install mkdocs-material mkdocstrings[python]

# Serve documentation locally (auto-refreshes on changes)
mkdocs serve

# Build static documentation site
mkdocs build
```

The documentation will be available at http://localhost:8000 when served locally.

## Key Features

1. **Markdown Support**: Documentation is written in familiar Markdown format
2. **Python Docstring Extraction**: Automatically pulls documentation from Python code
3. **Material Theme**: Clean, responsive design with dark/light mode
4. **Search Functionality**: Full-text search across all documentation
5. **API Documentation**: Complete API reference extracted from source code
6. **Tutorials**: Step-by-step guides for different use cases

## Deployment

The documentation is configured to be deployed to GitHub Pages when pushed to the `docs` branch. The GitHub Actions workflow in `.github/workflows/docs.yml` handles the deployment automatically.

## Contributing to Documentation

To add new documentation:

1. Create a new `.md` file in the appropriate directory
2. Add your content using Markdown syntax
3. Link to the new page in the `nav` section of `mkdocs.yml`
4. Submit a pull request

For API documentation, ensure your Python functions and classes have proper docstrings in Google, NumPy, or Sphinx format.