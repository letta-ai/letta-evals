"""Unit tests for HuggingFace Hub-backed datasets.

Covers URL detection/parsing, provenance extraction, the fetch wrapper (with a
stubbed ``huggingface_hub``), the end-to-end ``load_dataset`` path including
that ``rubric_path`` resolves against the suite dir (not the HF cache dir), and
that resolved provenance serializes into the suite config (suite.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import huggingface_hub
import pytest

from letta_evals.datasets.hf import (
    HfDatasetRef,
    _commit_sha_from_cache_path,
    hf_dataset_provenance,
    is_hf_ref,
    parse_hf_ref,
    resolve_hf_dataset,
)
from letta_evals.datasets.loader import load_dataset
from letta_evals.models import SuiteSpec


def _install_fake_hub(monkeypatch, download_path=None, files=None):
    """Patch huggingface_hub's fetch fns with in-memory stubs; return call log."""
    calls = {"download": [], "list": []}

    def _download(repo_id, filename, repo_type, revision):
        calls["download"].append(
            {"repo_id": repo_id, "filename": filename, "repo_type": repo_type, "revision": revision}
        )
        return str(download_path)

    def _list(repo_id, repo_type, revision):
        calls["list"].append({"repo_id": repo_id, "repo_type": repo_type, "revision": revision})
        return list(files or [])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", _list)
    return calls


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
            "http://huggingface.co/datasets/org/repo/blob/v1/d.csv",
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
            "https://huggingface.co/org/model/resolve/v1/d.jsonl",  # model repo, not a dataset
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
        assert ref == HfDatasetRef("letta-ai/swe-chat-tagged", "main", "train.jsonl")

    def test_blob_url(self):
        ref = parse_hf_ref("https://huggingface.co/datasets/org/repo/blob/v1.0/data.csv")
        assert ref == HfDatasetRef("org/repo", "v1.0", "data.csv")

    def test_bare_repo_url(self):
        ref = parse_hf_ref("https://huggingface.co/datasets/letta-ai/swe-chat-tagged")
        assert ref == HfDatasetRef("letta-ai/swe-chat-tagged", None, None)

    def test_nested_path(self):
        ref = parse_hf_ref("https://huggingface.co/datasets/org/repo/resolve/main/sub/dir/data.jsonl")
        assert ref.path == "sub/dir/data.jsonl"

    def test_non_dataset_url_raises(self):
        with pytest.raises(ValueError, match="expected"):
            parse_hf_ref("https://huggingface.co/org/model/resolve/v1/d.jsonl")

    def test_resolve_missing_file_raises(self):
        with pytest.raises(ValueError, match="missing a revision and/or file path"):
            parse_hf_ref("https://huggingface.co/datasets/org/repo/resolve/main")

    def test_single_segment_raises(self):
        with pytest.raises(ValueError, match="expected"):
            parse_hf_ref("https://huggingface.co/datasets/onlyorg")


# ── provenance helpers ─────────────────────────────────────────────────────


class TestCommitShaFromCachePath:
    def test_extracted(self):
        p = Path("/home/u/.cache/huggingface/datasets--org--repo/snapshots/deadbeef/data.jsonl")
        assert _commit_sha_from_cache_path(p) == "deadbeef"

    def test_none_when_no_snapshots(self):
        assert _commit_sha_from_cache_path(Path("/tmp/local/data.jsonl")) is None


# ── resolve_hf_dataset ─────────────────────────────────────────────────────


class TestResolveHfDataset:
    def test_file_url_returns_path_and_provenance(self, tmp_path, monkeypatch):
        cache = _cache_file(tmp_path, "abc123", "train.jsonl", [{"input": "Q?"}])
        calls = _install_fake_hub(monkeypatch, download_path=cache)

        resolved = resolve_hf_dataset("https://huggingface.co/datasets/org/repo/resolve/abc123/train.jsonl")

        assert resolved.local_path == cache
        assert resolved.commit_sha == "abc123"
        assert resolved.repo_id == "org/repo"
        assert resolved.path == "train.jsonl"
        assert calls["download"] == [
            {"repo_id": "org/repo", "filename": "train.jsonl", "repo_type": "dataset", "revision": "abc123"}
        ]

    def test_unpinned_revision_warns(self, tmp_path, monkeypatch):
        cache = _cache_file(tmp_path, "sha9", "d.jsonl", [{"input": "Q?"}])
        _install_fake_hub(monkeypatch, download_path=cache)

        with pytest.warns(UserWarning, match="unpinned"):
            resolve_hf_dataset("https://huggingface.co/datasets/org/repo/resolve/main/d.jsonl")

    def test_pinned_revision_does_not_warn(self, tmp_path, monkeypatch, recwarn):
        cache = _cache_file(tmp_path, "abc123", "d.jsonl", [{"input": "Q?"}])
        _install_fake_hub(monkeypatch, download_path=cache)

        resolve_hf_dataset("https://huggingface.co/datasets/org/repo/resolve/abc123/d.jsonl")
        assert not [w for w in recwarn.list if "unpinned" in str(w.message)]

    def test_bare_repo_single_manifest_resolved(self, tmp_path, monkeypatch):
        cache = _cache_file(tmp_path, "sha9", "the_only.jsonl", [{"input": "Q?"}])
        calls = _install_fake_hub(monkeypatch, download_path=cache, files=["README.md", "the_only.jsonl", "x.yaml"])

        with pytest.warns(UserWarning):  # bare repo => unpinned
            resolved = resolve_hf_dataset("https://huggingface.co/datasets/org/repo")

        assert resolved.path == "the_only.jsonl"
        assert calls["download"][0]["filename"] == "the_only.jsonl"

    def test_bare_repo_multiple_manifests_raises(self, tmp_path, monkeypatch):
        _install_fake_hub(monkeypatch, download_path=tmp_path / "x", files=["train.jsonl", "test.jsonl"])

        with pytest.raises(ValueError, match="multiple manifests"):
            resolve_hf_dataset("https://huggingface.co/datasets/org/repo")

    def test_bare_repo_no_manifest_raises(self, tmp_path, monkeypatch):
        _install_fake_hub(monkeypatch, download_path=tmp_path / "x", files=["README.md", "weights.bin"])

        with pytest.raises(ValueError, match="No .jsonl/.csv manifest"):
            resolve_hf_dataset("https://huggingface.co/datasets/org/repo")


# ── load_dataset end-to-end over an HF ref ─────────────────────────────────


class TestLoadDatasetHf:
    def test_loads_samples_from_hf_url(self, tmp_path, monkeypatch):
        cache = _cache_file(tmp_path, "abc", "d.jsonl", [{"input": "Q1"}, {"input": "Q2"}])
        _install_fake_hub(monkeypatch, download_path=cache)

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
        _install_fake_hub(monkeypatch, download_path=cache)

        samples = list(
            load_dataset(
                "https://huggingface.co/datasets/org/repo/resolve/abc/d.jsonl",
                base_dir=suite_dir,
            )
        )
        assert samples[0].rubric == "suite-level rubric"


# ── provenance record + suite.json serialization ───────────────────────────


def _suite_yaml(dataset: str) -> dict:
    return {
        "name": "hf-suite",
        "dataset": dataset,
        "target": {"kind": "letta_code", "model_handles": ["openai/gpt-4.1-mini"]},
        "graders": {"g": {"kind": "tool", "function": "exact_match"}},
        "reward": {"kind": "metric", "metric_key": "g"},
    }


class TestProvenance:
    def test_local_dataset_has_no_provenance(self):
        assert hf_dataset_provenance("data/local.jsonl") is None

    def test_hf_dataset_provenance_record(self, tmp_path, monkeypatch):
        cache = _cache_file(tmp_path, "abc123", "train.jsonl", [{"input": "Q?"}])
        _install_fake_hub(monkeypatch, download_path=cache)

        prov = hf_dataset_provenance("https://huggingface.co/datasets/org/repo/resolve/abc123/train.jsonl")
        assert prov == {
            "source": "huggingface",
            "url": "https://huggingface.co/datasets/org/repo/resolve/abc123/train.jsonl",
            "repo_id": "org/repo",
            "path": "train.jsonl",
            "revision": "abc123",
            "commit_sha": "abc123",
        }

    def test_provenance_serializes_into_suite_config(self):
        # Mirrors StreamingWriter.initialize, which dumps the suite into suite.json.
        suite = SuiteSpec.from_yaml(_suite_yaml("https://huggingface.co/datasets/org/repo/resolve/abc/d.jsonl"))
        suite.dataset_provenance = {"source": "huggingface", "commit_sha": "abc"}
        dumped = json.loads(suite.model_dump_json(exclude={"base_dir"}))
        assert dumped["dataset_provenance"]["commit_sha"] == "abc"


class TestFromYamlDataset:
    def test_hf_url_not_anchored_to_base_dir(self, tmp_path):
        url = "https://huggingface.co/datasets/org/repo/resolve/main/d.jsonl"
        suite = SuiteSpec.from_yaml(_suite_yaml(url), base_dir=tmp_path)
        assert suite.dataset == url  # unchanged, not joined to base_dir

    def test_local_relative_path_still_anchored(self, tmp_path):
        suite = SuiteSpec.from_yaml(_suite_yaml("data/d.jsonl"), base_dir=tmp_path)
        assert suite.dataset == str((tmp_path / "data/d.jsonl").resolve())
