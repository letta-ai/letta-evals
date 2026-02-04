---
name: Docker Build Optimization
description: Build smaller and faster container images by using layer-friendly Dockerfiles, caching, multi-stage builds, and deterministic dependency installs.
license: Proprietary. LICENSE.txt has complete terms
---

# Docker Build Optimization

## Overview

Docker builds are slow when the cache is constantly invalidated and images are large when build tools and source history leak into the runtime stage. The goal is to:
- Maximize layer cache hits
- Minimize what enters the final image
- Keep builds deterministic across machines/CI

## Principles

- **Cache key = instruction + filesystem state** at that layer.
- Put **rarely-changing steps first** (base image, system deps), then deps, then app code.
- Minimize the build context with **`.dockerignore`**.
- Use **multi-stage** so the runtime image contains only what’s needed to run.

## Workflow

1. **Check build context**
   - Add `.dockerignore` for `node_modules/`, `dist/`, `.git/`, caches, secrets, test artifacts.
2. **Reorder for caching**
   - Copy only dependency manifests first (e.g., `package.json` / `poetry.lock`), install deps, then copy the rest.
3. **Make installs deterministic**
   - Pin versions/lockfiles.
   - Prefer `npm ci`, `pip install -r requirements.txt --require-hashes`, `poetry install --sync`.
4. **Use multi-stage**
   - Build stage: compilers, dev deps, tests.
   - Runtime stage: only built artifacts + runtime deps.
5. **Measure**
   - Build with `--progress=plain` and look for steps that always rebuild.
   - Inspect layers: `docker history <image>`.

## Common Pitfalls

- `COPY . .` before installing deps → invalidates cache on any source change.
- Not cleaning package manager caches (`apt` lists, build caches).
- Shipping build tools (gcc, node-gyp, git) in runtime image.
- Leaking secrets into layers (baked into the image history).
- Huge contexts (sending gigabytes to the daemon).

## Example Pattern (Node-ish)

- Copy `package.json` + lockfile
- Install deps
- Copy source
- Build
- Copy `dist/` into a slim runtime stage

## Checklist

- [ ] `.dockerignore` exists and excludes large/secret dirs
- [ ] Dependency install happens before copying full source
- [ ] Uses lockfile-driven install (`npm ci`, etc.)
- [ ] Multi-stage separates build vs runtime
- [ ] Runtime image runs as non-root when possible
- [ ] Image size and build time are measured and tracked
