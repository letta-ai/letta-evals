---
name: Log Privacy & Redaction
description: Avoid leaking secrets and PII in logs while keeping them useful: structured logging, safe fields, redaction patterns, and incident-safe practices.
license: Proprietary. LICENSE.txt has complete terms
---

# Log Privacy & Redaction

## Overview

Logs are durable and widely accessible. Treat them as sensitive data. The goal is to:
- Preserve debuggability (correlation IDs, event types, error codes)
- Avoid sensitive exposure (tokens, passwords, personal data)
- Make redaction consistent and testable

## What to Log (Safe Defaults)

- Request ID / trace ID / correlation ID
- High-level event type (e.g., `payment_failed`)
- Error class + stable error code
- Timing/latency metrics
- Sanitized identifiers (hashes) instead of raw PII where possible

## What NOT to Log

- Access tokens / refresh tokens / API keys
- Passwords, secrets, session cookies
- Full credit card numbers, SSNs, full addresses
- Raw request bodies unless explicitly scrubbed

## Workflow

1. **Define a safe schema** for log fields (allowlist).
2. **Redact at boundaries**:
   - Incoming headers (Authorization)
   - Query params (tokens)
   - Error messages that echo input
3. **Redaction rules**:
   - Replace values with `[REDACTED:<TYPE>]`
   - Keep minimal prefixes/suffixes only if needed for debugging
4. **Test redaction**:
   - Unit tests for known patterns
   - Property tests that no secret patterns pass through
5. **Incident mode**:
   - Never “temporarily log everything”
   - Add targeted, short-lived instrumentation with explicit cleanup

## Common Patterns to Redact

- `Authorization: Bearer <...>`
- JWTs (`eyJ...`)
- AWS keys (`AKIA...`)
- Database URLs with credentials (`postgres://user:pass@...`)

## Checklist

- [ ] Allowlist schema for log fields
- [ ] Centralized redaction middleware
- [ ] Tests for redaction patterns
- [ ] No raw request/response bodies in logs by default
- [ ] Correlation IDs present for debugging
