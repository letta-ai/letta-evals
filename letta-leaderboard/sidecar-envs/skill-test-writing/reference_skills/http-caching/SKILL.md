---
name: HTTP Caching & Validation
description: Design and debug HTTP caching behavior using Cache-Control, ETag/If-None-Match, Last-Modified, CDN caches, and safe invalidation patterns.
license: Proprietary. LICENSE.txt has complete terms
---

# HTTP Caching & Validation

## Overview

Caching is a correctness feature, not just performance. The same endpoint can behave differently depending on:
- Response headers (`Cache-Control`, `ETag`, `Vary`, `Set-Cookie`)
- Client and CDN cache behavior
- URL normalization and query params

## Core Concepts

- **Freshness**: `Cache-Control: max-age=...` controls how long a cached response is considered fresh.
- **Validation**: `ETag` + `If-None-Match` enables revalidation (304 Not Modified).
- **Shared vs private**: `public` vs `private` determines whether CDNs/browsers may share cached responses.
- **Vary**: describes which request headers affect the response (e.g., `Vary: Accept-Encoding, Authorization`).

## Workflow: Debug a “Stale Content” Bug

1. Confirm the cache layer: browser cache, CDN, reverse proxy, application.
2. Capture request/response headers (including `Age`, `Via`, `X-Cache` if present).
3. Check cache-busting inputs:
   - URL path and query params
   - `Vary` headers
   - Authorization/cookies (often imply `private` caching)
4. Verify correctness headers:
   - For user-specific content: `Cache-Control: private, no-store` (often safest)
   - For static assets: long `max-age` + immutable filenames
5. Add validation:
   - Provide `ETag` or `Last-Modified` for resources that change.
6. Invalidation strategy:
   - Prefer **versioned URLs** for static assets.
   - Avoid purges as the only mechanism (slow and error-prone).

## Common Pitfalls

- Caching responses that include `Set-Cookie` unintentionally.
- Missing `Vary: Authorization` on endpoints that return user-specific content.
- Using long `max-age` on mutable URLs (no versioning) → hard-to-debug staleness.
- Assuming 304 means “no work happened” (it still requires revalidation request).

## Checklist

- [ ] Is content user-specific? If yes, ensure private/no-store.
- [ ] Are assets versioned (hash in filename) if using long max-age?
- [ ] Is `Vary` correct for headers that change response?
- [ ] Do you provide `ETag`/`Last-Modified` for mutable resources?
- [ ] Can you reproduce with CDN bypass and with empty browser cache?
