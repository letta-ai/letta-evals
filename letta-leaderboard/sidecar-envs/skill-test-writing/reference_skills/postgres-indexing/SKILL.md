---
name: PostgreSQL Indexing & EXPLAIN
description: Use EXPLAIN ANALYZE to find bottlenecks and choose the right indexes (btree/gin/partial/covering) with safe production rollout practices.
license: Proprietary. LICENSE.txt has complete terms
---

# PostgreSQL Indexing & EXPLAIN

## Overview

Indexing is about matching your query patterns. A “missing index” isn’t always the issue — sometimes the query shape, statistics, or join strategy is the bottleneck. Use EXPLAIN ANALYZE to ground decisions in evidence.

## Workflow

1. Capture the slow query + parameters.
2. Run `EXPLAIN (ANALYZE, BUFFERS)` in a safe environment (or on prod for read-only queries).
3. Identify the primary cost center:
   - Full table scans (Seq Scan)
   - Large sorts
   - Nested loops with huge row counts
   - Hash joins spilling to disk
4. Decide the fix:
   - Add/adjust indexes
   - Rewrite query (predicate pushdown, reduce rows early)
   - Refresh stats (`ANALYZE`)
5. Roll out safely:
   - Use `CREATE INDEX CONCURRENTLY` in production to avoid blocking writes.
   - Monitor build time and IO impact.

## Index Selection Cheatsheet

- **B-tree**: equality + range filters, ordering, most common.
- **Composite index**: when queries filter on multiple columns; order columns by selectivity and query patterns.
- **Partial index**: index only rows that match a predicate (e.g., `WHERE deleted_at IS NULL`).
- **Covering index**: include extra columns to avoid heap fetches (via `INCLUDE`).
- **GIN**: arrays/JSONB/text search.

## Pitfalls

- Low-selectivity columns (e.g., boolean) rarely benefit alone.
- Index doesn’t help if predicate doesn’t match index order.
- Too many indexes slow writes and bloat storage.

## Checklist

- [ ] Use EXPLAIN ANALYZE to confirm bottleneck
- [ ] Choose index type based on predicate and data shape
- [ ] Prefer CONCURRENTLY for prod
- [ ] Measure before/after and watch write overhead
