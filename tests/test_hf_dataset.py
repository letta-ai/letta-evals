"""Unit tests for HuggingFace Hub-backed datasets.

Covers URL detection/parsing, provenance extraction, the fetch wrapper (with a
stubbed ``huggingface_hub``), and the end-to-end ``load_dataset`` path including
that ``rubric_path`` resolves against the suite dir, not the HF cache dir.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from letta_evals.datasets import hf
from letta_evals.datasets.hf import (
    HfDatasetRef,
    _commit_sha_from_cache_path,
    _import_hf,
    _is_unpinned,
    is_hf_ref,
    parse_hf_ref,
    resolve_hf_dataset,
)
from letta_evals.datasets.loader import load_dataset
from letta_evals.models import SuiteSpec


class _FakeHub:
    """Stand-in for the ``huggingface_hub`` module in tests."""

    def __init__(self, download_path, files=None):
        self.download_path = str(download_path)
        self.files = list(files) if files is not None else []
        self.download_calls = []

    def hf_hub_download(self, repo_id, filename, repo_type, revision):
        self.download_calls.append(
            {"repo_id": repo_id, "filename": filename, "repo_type": repo_type, "revision": revision}
        )
        return self.download_path

    def list_repo_files(self, repo_id, repo_type, revision):
        return list(self.files)


def _cache_file(tmp_path: Path, sha: str, filename: str, rows) -> Path:
    """Materialize a fake hf cache file at <tmp>/…/snapshots/<sha>/<filename>."""
    path = tmp_path / "datasets--org--repo" / "snapshots" / sha / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


# ── is_hf_ref ──────────────────────────────────────────────────────────────


class TestIsHfRef:
    @pytest.mark.parametrize(
        "value",
        [
            "https://huggingface.co/datasets/org/repo",
            "https://huggingface.co/datasets/org/repo/resolve/main/d.jsonl",
            "http://huggingface.co/org/model/resolve/v1/d.jsonl",
            "HTTPS://HuggingFace.co/datasets/org/repo",  # case-insensitive host/scheme
        ],
    )
    def test_true(self, value):
        assert is_hf_ref(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "data.jsonl",
            "/abs/path/data.jsonl",
            "datasets/local.jsonl",
            "https://example.com/data.jsonl",
            "https://huggingface.co.evil.com/datasets/org/repo",  # look-alike host
            Path("data.jsonl"),
            None,
        ],
    )
    def test_false(self, value):
        assert is_hf_ref(value) is False


# ── parse_hf_ref ───────────────────────────────────────────────────────────


class TestParseHfRef:
    def test_resolve_url(self):
        ref = parse_hf_ref("https://huggingface.co/datasets/letta-ai/swe-chat-tagged/resolve/main/train.jsonl")
        assert ref == HfDatasetRef("letta-ai/swe-chat-tagged", "dataset", "main", "train.jsonl")

    def test_blob_url(self):
        ref = parse_hf_ref("https://huggingface.co/datasets/org/repo/blob/v1.0/data.csv")
        assert ref == HfDatasetRef("org/repo", "dataset", "v1.0", "data.csv")

    def test_bare_repo_url(self):
        ref = parse_hf_ref("https://huggingface.co/datasets/letta-ai/swe-chat-tagged")
        assert ref == HfDatasetRef("letta-ai/swe-chat-tagged", "dataset", None, None)

    def test_model_repo(self):
        ref = parse_hf_ref("https://huggingface.co/org/model/resolve/abc123/d.jsonl")
        assert ref == HfDatasetRef("org/model", "model", "abc123", "d.jsonl")

    def test_space_repo(self):
        ref = parse_hf_ref("https://huggingface.co/spaces/org/space/resolve/main/d.csv")
        assert ref.repo_type == "space"
        assert ref.repo_id == "org/space"

    def test_nested_path(self):
        ref = parse_hf_ref("https://huggingface.co/datasets/org/repo/resolve/main/sub/dir/data.jsonl")
        assert ref.path == "sub/dir/data.jsonl"

    def test_resolve_missing_file_raises(self):
        with pytest.raises(ValueError, match="missing a revision and/or file path"):
            parse_hf_ref("https://huggingface.co/datasets/org/repo/resolve/main")

    def test_single_segment_raises(self):
        with pytest.raises(ValueError, match="expected"):
            parse_hf_ref("https://huggingface.co/datasets/onlyorg")


# ── provenance helpers ─────────────────────────────────────────────────────


class TestProvenanceHelpers:
    def test_commit_sha_extracted(self):
        p = Path("/home/u/.cache/huggingface/datasets--org--repo/snapshots/deadbeef/data.jsonl")
        assert _commit_sha_from_cache_path(p) == "deadbeef"

    def test_commit_sha_none_when_no_snapshots(self):
        assert _commit_sha_from_cache_path(Path("/tmp/local/data.jsonl")) is None

    @pytest.mark.parametrize("rev,expected", [(None, True), ("main", True), ("master", True)])
    def test_unpinned(self, rev, expected):
        assert _is_unpinned(rev) is expected

    @pytest.mark.parametrize("rev", ["abc123def", "v1.0", "release-2024"])
    def test_pinned(self, rev):
        assert _is_unpinned(rev) is False


# ── _import_hf ─────────────────────────────────────────────────────────────


def test_import_hf_missing_raises():
    if importlib.util.find_spec("huggingface_hub") is not None:
        pytest.skip("huggingface_hub is installed; cannot exercise the missing-dep path")
    with pytest.raises(ImportError, match=r"letta-evals\[hf\]"):
        _import_hf()


# ── resolve_hf_dataset ─────────────────────────────────────────────────────


class TestResolveHfDataset:
    def test_file_url_returns_path_and_provenance(self, tmp_path, monkeypatch):
        cache = _cache_file(tmp_path, "abc123", "train.jsonl", [{"input": "Q?"}])
        fake = _FakeHub(cache)
        monkeypatch.setattr(hf, "_import_hf", lambda: fake)

        resolved = resolve_hf_dataset("https://huggingface.co/datasets/org/repo/resolve/abc123/train.jsonl")

        assert resolved.local_path == cache
        assert resolved.commit_sha == "abc123"
        assert resolved.repo_id == "org/repo"
        assert resolved.path == "train.jsonl"
        assert fake.download_calls == [
            {"repo_id": "org/repo", "filename": "train.jsonl", "repo_type": "dataset", "revision": "abc123"}
        ]

    def test_unpinned_revision_warns(self, tmp_path, monkeypatch):
        cache = _cache_file(tmp_path, "sha9", "d.jsonl", [{"input": "Q?"}])
        monkeypatch.setattr(hf, "_import_hf", lambda: _FakeHub(cache))

        with pytest.warns(UserWarning, match="unpinned"):
            resolve_hf_dataset("https://huggingface.co/datasets/org/repo/resolve/main/d.jsonl")

    def test_bare_repo_single_manifest_resolved(self, tmp_path, monkeypatch):
        cache = _cache_file(tmp_path, "sha9", "the_only.jsonl", [{"input": "Q?"}])
        fake = _FakeHub(cache, files=["README.md", "the_only.jsonl", "config.yaml"])
        monkeypatch.setattr(hf, "_import_hf", lambda: fake)

        with pytest.warns(UserWarning):  # bare repo => unpinned
            resolved = resolve_hf_dataset("https://huggingface.co/datasets/org/repo")

        assert resolved.path == "the_only.jsonl"
        assert fake.download_calls[0]["filename"] == "the_only.jsonl"

    def test_bare_repo_multiple_manifests_raises(self, tmp_path, monkeypatch):
        fake = _FakeHub(tmp_path / "x", files=["train.jsonl", "test.jsonl"])
        monkeypatch.setattr(hf, "_import_hf", lambda: fake)

        with pytest.raises(ValueError, match="multiple manifests"):
            resolve_hf_dataset("https://huggingface.co/datasets/org/repo")

    def test_bare_repo_no_manifest_raises(self, tmp_path, monkeypatch):
        fake = _FakeHub(tmp_path / "x", files=["README.md", "weights.bin"])
        monkeypatch.setattr(hf, "_import_hf", lambda: fake)

        with pytest.raises(ValueError, match="No .jsonl/.csv manifest"):
            resolve_hf_dataset("https://huggingface.co/datasets/org/repo")


# ── load_dataset end-to-end over an HF ref ─────────────────────────────────


class TestLoadDatasetHf:
    def test_loads_samples_from_hf_url(self, tmp_path, monkeypatch):
        cache = _cache_file(tmp_path, "abc", "d.jsonl", [{"input": "Q1"}, {"input": "Q2"}])
        monkeypatch.setattr(hf, "_import_hf", lambda: _FakeHub(cache))

        samples = list(load_dataset("https://huggingface.co/datasets/org/repo/resolve/abc/d.jsonl"))
        assert [s.input for s in samples] == ["Q1", "Q2"]

    def test_rubric_path_resolves_against_suite_dir_not_cache(self, tmp_path, monkeypatch):
        # Rubric file lives next to the suite; the manifest lives in the HF
        # cache. Relative rubric_path must resolve against base_dir (the suite),
        # never the cache dir where the fetched manifest happens to land.
        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()
        (suite_dir / "rubric.txt").write_text("suite-level rubric")
        cache = _cache_file(tmp_path, "abc", "d.jsonl", [{"input": "Q?", "rubric_path": "rubric.txt"}])
        monkeypatch.setattr(hf, "_import_hf", lambda: _FakeHub(cache))

        samples = list(
            load_dataset(
                "https://huggingface.co/datasets/org/repo/resolve/abc/d.jsonl",
                base_dir=suite_dir,
            )
        )
        assert samples[0].rubric == "suite-level rubric"


# ── SuiteSpec.from_yaml leaves HF URLs untouched ───────────────────────────


def _suite_yaml(dataset: str) -> dict:
    return {
        "name": "hf-suite",
        "dataset": dataset,
        "target": {"kind": "letta_code", "model_handles": ["openai/gpt-4.1-mini"]},
        "graders": {"g": {"kind": "tool", "function": "exact_match"}},
        "reward": {"kind": "metric", "metric_key": "g"},
    }


class TestFromYamlDataset:
    def test_hf_url_not_anchored_to_base_dir(self, tmp_path):
        url = "https://huggingface.co/datasets/org/repo/resolve/main/d.jsonl"
        suite = SuiteSpec.from_yaml(_suite_yaml(url), base_dir=tmp_path)
        assert suite.dataset == url  # unchanged, not joined to base_dir

    def test_local_relative_path_still_anchored(self, tmp_path):
        suite = SuiteSpec.from_yaml(_suite_yaml("data/d.jsonl"), base_dir=tmp_path)
        assert suite.dataset == str((tmp_path / "data/d.jsonl").resolve())
