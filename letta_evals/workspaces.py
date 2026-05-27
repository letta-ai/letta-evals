"""Workspace helpers for per-sample eval isolation."""

from __future__ import annotations

import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

from letta_evals.models import GitWorktreeSpec, SampleId


def _slug(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-._")
    return text[:80] or "sample"


class GitWorktreeWorkspace:
    """Create git worktrees for a single sample/model run."""

    def __init__(self, spec: GitWorktreeSpec, parent_root: Path, run_name: str, sample_id: SampleId, model_name: str | None):
        self.spec = spec
        self.parent_root = parent_root
        self.run_name = run_name
        self.sample_id = sample_id
        self.model_name = model_name or "model"
        sample_dir = f"sample-{_slug(sample_id)}"
        self.root = parent_root / "worktrees" / run_name / sample_dir
        self.artifact_dir = parent_root / "artifacts" / run_name / sample_dir

    def create(self) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=False)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        repos = self._normalize_repos()
        checkouts: dict[str, str] = {}
        for name, source in repos.items():
            checkout = self.root / name
            self._add_worktree(source, checkout)
            checkouts[name] = str(checkout)
        working_dir = next(iter(checkouts.values())) if len(repos) == 1 else str(self.root)
        return {
            "kind": "git_worktree",
            "root": str(self.root),
            "working_dir": working_dir,
            "artifact_dir": str(self.artifact_dir),
            "repos": {name: str(source) for name, source in repos.items()},
            "checkouts": checkouts,
        }

    def _normalize_repos(self) -> dict[str, Path]:
        if self.spec.repo is not None:
            source = Path(self.spec.repo).expanduser().resolve()
            return {source.name: source}
        raw = self.spec.repos
        if isinstance(raw, dict):
            return {str(name): Path(path).expanduser().resolve() for name, path in raw.items()}
        if isinstance(raw, list):
            paths = [Path(path).expanduser().resolve() for path in raw]
            return {path.name: path for path in paths}
        raise ValueError("git_worktree.repos must be a list or mapping")

    @staticmethod
    def _add_worktree(source: Path, checkout: Path) -> None:
        if not source.exists():
            raise ValueError(f"git_worktree repo does not exist: {source}")
        subprocess.run(
            ["git", "-C", str(source), "worktree", "add", "--detach", str(checkout), "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
