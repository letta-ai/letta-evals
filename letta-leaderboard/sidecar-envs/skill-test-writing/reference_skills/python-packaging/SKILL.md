---
name: Python Packaging (pyproject)
description: Package Python projects correctly with pyproject.toml, build wheels, manage editable installs, and avoid common dependency/versioning pitfalls.
license: Proprietary. LICENSE.txt has complete terms
---

# Python Packaging (pyproject)

## Overview

Modern Python packaging centers on `pyproject.toml`. The common goals are:
- Reproducible installs (dev vs CI)
- Clear build backend configuration
- Correct versioning and distribution artifacts (sdist/wheel)

## Workflow

1. Define project metadata in `pyproject.toml`:
   - name, version, dependencies, optional extras
2. Choose a build backend (e.g., setuptools, hatchling, poetry-core).
3. For development:
   - Use a virtualenv
   - Install editable: `pip install -e .` (if supported by backend)
4. For release:
   - Build: `python -m build`
   - Verify artifacts: install the wheel into a clean env and run imports/tests
5. Pin dependencies appropriately:
   - Use lockfiles for apps, more flexible ranges for libraries

## Common Pitfalls

- Confusing an app vs a library: apps can lock tightly; libraries should be compatible across versions.
- Editable install surprises with PEP 660: backend must support it.
- Import/package name mismatch: distribution name can differ from module import name.
- Missing package data: ensure non-.py files are included in the build.

## Checklist

- [ ] `pyproject.toml` defines metadata and dependencies
- [ ] Clean build produces wheel + sdist
- [ ] Wheel installs and imports in a fresh environment
- [ ] Versioning strategy is consistent (tags, changelog, semver)
