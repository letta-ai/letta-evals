---
name: OAuth/OIDC Practical Debugging
description: Diagnose common OAuth2/OIDC failures in authorization code + PKCE flows, validate tokens correctly, and debug production-only auth issues safely.
license: Proprietary. LICENSE.txt has complete terms
---

# OAuth/OIDC Practical Debugging

## Overview

Most OAuth/OIDC issues come from mismatched inputs between steps (redirect URI, PKCE verifier, client settings) or incorrect token validation. The fastest path is to trace the full flow and compare exact values.

## Flow Checklist (Auth Code + PKCE)

1. **/authorize request**
   - `client_id`, `redirect_uri`, `scope`, `response_type=code`
   - Generate `state` (CSRF) and `nonce` (OIDC) and persist them
   - Generate `code_verifier` and derived `code_challenge`
2. **Redirect callback**
   - Verify `state`
   - Capture `code`
3. **/token exchange**
   - Send the same `redirect_uri`
   - Send `code_verifier` for PKCE
4. **Validate tokens**
   - Access token: used to call APIs (often opaque)
   - ID token: validate signature + claims (`iss`, `aud`, `exp`, `nonce`)

## Debugging invalid_grant

Common causes:
- Redirect URI mismatch (must match exactly)
- Code expired or reused
- Wrong `code_verifier` (lost between steps or wrong user/session)
- Clock skew when server enforces narrow time windows

Actionable checks:
- Compare the exact `redirect_uri` string from /authorize and /token
- Confirm the code is exchanged once
- Confirm verifier stored per user session and retrieved correctly

## Token Validation Pitfalls

- Validate `iss` and `aud` against the correct environment (dev vs prod tenant).
- Cache JWKS but handle key rotation (respect cache headers; retry on kid mismatch).
- Never accept an ID token without verifying signature and `exp`.

## Safe Logging

Never log:
- auth codes, access tokens, refresh tokens, client secrets

Log instead:
- request IDs, correlation IDs, error codes, and redacted token fingerprints

## Checklist

- [ ] Trace /authorize → callback → /token end-to-end
- [ ] Compare redirect_uri and PKCE verifier values
- [ ] Validate ID token signature + iss/aud/exp/nonce
- [ ] Check env-specific settings (issuer, allowed origins, redirect URIs)
