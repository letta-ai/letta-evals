---
name: Git History Surgery
description: Safely rewrite, clean up, and recover Git history using rebase/cherry-pick/revert and reflog without breaking shared branches.
license: Proprietary. LICENSE.txt has complete terms
---

# Git History Surgery

## Overview

Rewriting Git history is powerful and dangerous. The core safety rule is:

**Do not rewrite commits that other people have based work on.**

If a commit is already on a shared branch (e.g., `main`), prefer **revert** over rebase/reset.

## Decision Tree

- Need to fix the last commit message/content? → `git commit --amend`
- Need to reorder/squash/drop commits on a branch not yet shared? → `git rebase -i`
- Need to move specific commits between branches? → `git cherry-pick`
- Need to undo a bad merge/rewrite? → `git reflog` + `git reset --hard`

## Workflow: Interactive Rebase (Safe Version)

1. Ensure clean status: `git status` (no uncommitted changes).
2. Create a backup pointer: `git branch backup/<name>`.
3. Start interactive rebase: `git rebase -i <base>`.
4. Use actions intentionally:
   - `reword`: change message
   - `edit`: modify content
   - `squash`/`fixup`: combine commits
   - `drop`: remove a commit
5. Resolve conflicts:
   - Fix files, `git add`, then `git rebase --continue`.
6. Run tests and sanity checks.
7. If the branch was pushed before, you’ll likely need:
   - `git push --force-with-lease` (safer than `--force`).

## Recovery Patterns

- Abort an in-progress rebase: `git rebase --abort`
- Find “where things were”: `git reflog`
- Restore the branch: `git reset --hard <reflog_hash>`
- Recover a dropped commit: find it in reflog and `git cherry-pick <hash>`

## Common Pitfalls

- **Force pushing over someone else’s work**: use `--force-with-lease` and coordinate.
- **Rebasing merges incorrectly**: if you need to preserve merges, consider `--rebase-merges`.
- **Losing local changes**: make a backup branch or stash before surgery.
- **Accidentally rewriting `main`**: avoid history rewrites on protected branches.

## Checklist

- [ ] Confirm the branch is safe to rewrite (not shared)
- [ ] Create a backup branch
- [ ] Use `--force-with-lease` if pushing rewritten history
- [ ] Use reflog for recovery
- [ ] Run tests after rewriting
