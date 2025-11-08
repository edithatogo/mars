# pymars Repository Branch Structure

## Overview

This document explains the branch structure of the pymars repository, which separates the software implementation from the research paper.

## Branches

### `main` branch
- Contains the core pymars software library
- Includes source code, tests, documentation, and examples
- This is the primary branch for software development
- Used for releases and day-to-day development

### `paper` branch  
- Contains the research paper describing pymars and its applications
- Includes LaTeX source files, compiled PDFs, and related paper materials
- Separated to keep the main repository focused on the software
- Contains examples specific to health economic applications

## Benefits of This Structure

1. **Separation of Concerns**: Software development and paper writing happen in separate branches
2. **Reduced Repository Size**: Main branch contains only the essential software components
3. **Focused Development**: Contributors can focus on the software without paper-related files
4. **Independent Workflows**: Paper review and revision can happen independently of software development
5. **Clean History**: Each branch maintains a focused commit history relevant to its purpose

## How to Access Each Component

### To work with the software:
```bash
git clone https://github.com/edithatogo/pymars.git
cd pymars  # You're now on main branch by default
```

### To access the paper:
```bash
git clone https://github.com/edithatogo/pymars.git
cd pymars
git checkout paper  # Switch to the paper branch
```

## Maintaining the Structure

- Software improvements and bug fixes go in the `main` branch
- Paper revisions and additions go in the `paper` branch
- Feature development for new capabilities should generally happen in `main` first
- When new features are stable and documented, they can be showcased in paper examples in the `paper` branch