"""HuggingFace Hub-backed dataset resolution.

Resolves an HF dataset reference in a suite's ``dataset:`` field to a local
file, which the loader then reads exactly as it would a local path. Only the
single manifest file named in the reference is fetched -- never the whole repo
-- so large repos stay cheap, and on the Modal sandbox path nothing is fetched
in-sandbox (the host loads the dataset and ships each sample as JSON via
``--sample``).

``huggingface_hub`` is imported lazily -- only when an HF ref is actually used
-- so suites that only reference local paths don't pay for the import at
startup.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

_HF_HOST = "huggingface.co"
_MANIFEST_SUFFIXES = (".jsonl", ".csv")


def is_hf_ref(value) -> bool:
    """Return True if ``value`` is a HuggingFace Hub *dataset* URL.

    Only ``https://huggingface.co/datasets/...`` (and the http variant) count;
    everything else -- local path strings and ``Path`` objects -- is left
    untouched so filesystem datasets behave exactly as before.
    """
    if not isinstance(value, str):
        return False
    v = value.strip().lower()
    return v.startswith(f"https://{_HF_HOST}/datasets/") or v.startswith(f"http://{_HF_HOST}/datasets/")


@dataclass(frozen=True)
class HfDatasetRef:
    """A parsed HuggingFace Hub reference to a single dataset manifest file."""

    repo_id: str
    revision: Optional[str]  # None => unpinned; resolves to the repo default branch
    path: Optional[str]  # None => bare repo URL; the manifest is resolved on fetch


@dataclass(frozen=True)
class ResolvedHfDataset:
    """The outcome of fetching an ``HfDatasetRef`` to local disk."""

    local_path: Path
    repo_id: str
    revision: Optional[str]
    path: str
    commit_sha: Optional[str]  # the exact commit the run read, when recoverable


def parse_hf_ref(value: str) -> HfDatasetRef:
    """Parse a huggingface.co dataset URL into an :class:`HfDatasetRef`.

    Accepts the repo landing URL and the file (``/resolve/`` or ``/blob/``)
    URL::

        https://huggingface.co/datasets/<org>/<repo>
        https://huggingface.co/datasets/<org>/<repo>/resolve/<rev>/<path>
        https://huggingface.co/datasets/<org>/<repo>/blob/<rev>/<path>
    """
    parsed = urlparse(value.strip())
    segments = [unquote(s) for s in parsed.path.split("/") if s]

    if not segments or segments[0] != "datasets":
        raise ValueError(
            f"Cannot parse HuggingFace dataset URL {value!r}: expected "
            "'https://huggingface.co/datasets/<org>/<repo>[/resolve/<rev>/<file>]'."
        )
    segments = segments[1:]

    if len(segments) < 2:
        raise ValueError(
            f"Cannot parse HuggingFace dataset URL {value!r}: expected "
            "'https://huggingface.co/datasets/<org>/<repo>[/resolve/<rev>/<file>]'."
        )

    repo_id = f"{segments[0]}/{segments[1]}"
    rest = segments[2:]

    if not rest:
        # Bare repo URL: manifest file and revision are resolved on fetch.
        return HfDatasetRef(repo_id=repo_id, revision=None, path=None)

    if rest[0] in ("resolve", "blob"):
        if len(rest) < 3:
            raise ValueError(
                f"HuggingFace file URL {value!r} is missing a revision and/or file path; "
                "expected '.../resolve/<rev>/<file>'."
            )
        return HfDatasetRef(repo_id=repo_id, revision=rest[1], path="/".join(rest[2:]))

    raise ValueError(
        f"Unrecognized HuggingFace URL shape {value!r}; expected a repo URL or a "
        "'.../resolve/<rev>/<file>' file URL."
    )


def _commit_sha_from_cache_path(local_path: Path) -> Optional[str]:
    """Extract the resolved commit SHA from an ``hf_hub_download`` cache path.

    Downloads resolve under ``<cache>/<repo>/snapshots/<commit_sha>/<file>``;
    the segment right after ``snapshots`` is the exact commit the run read.
    Returns ``None`` if the layout isn't recognized (best-effort provenance).
    """
    parts = local_path.parts
    try:
        i = parts.index("snapshots")
    except ValueError:
        return None
    return parts[i + 1] if i + 1 < len(parts) else None


def _resolve_manifest_filename(ref: HfDatasetRef) -> str:
    """Find the single ``.jsonl``/``.csv`` manifest in a bare-repo reference."""
    import huggingface_hub

    files = huggingface_hub.list_repo_files(repo_id=ref.repo_id, repo_type="dataset", revision=ref.revision)
    candidates = [f for f in files if f.lower().endswith(_MANIFEST_SUFFIXES)]
    if not candidates:
        raise ValueError(
            f"No .jsonl/.csv manifest found in HuggingFace repo {ref.repo_id!r}. "
            "Point 'dataset:' at the file directly, e.g. "
            f"https://huggingface.co/datasets/{ref.repo_id}/resolve/<rev>/<file>."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"HuggingFace repo {ref.repo_id!r} has multiple manifests {sorted(candidates)}; "
            "point 'dataset:' at one of them, e.g. "
            f"https://huggingface.co/datasets/{ref.repo_id}/resolve/<rev>/<file>."
        )
    return candidates[0]


def resolve_hf_dataset(value: str) -> ResolvedHfDataset:
    """Fetch the manifest named by an HF URL and return its local path + provenance.

    Uses ``huggingface_hub``'s local cache, so repeated runs and repeated
    samples within a run don't re-download. The token is read from the standard
    ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` env (``huggingface_hub``'s
    default), so private repos work with no extra config. Warns when the
    revision is unpinned (mutable), surfacing the resolved commit SHA so the run
    stays reproducible from its logs.
    """
    import huggingface_hub

    ref = parse_hf_ref(value)
    filename = ref.path if ref.path is not None else _resolve_manifest_filename(ref)

    local_path = Path(
        huggingface_hub.hf_hub_download(
            repo_id=ref.repo_id,
            filename=filename,
            repo_type="dataset",
            revision=ref.revision,
        )
    )
    commit_sha = _commit_sha_from_cache_path(local_path)

    if ref.revision is None or ref.revision in ("main", "master"):
        pinned = commit_sha or "<commit-sha>"
        warnings.warn(
            f"HuggingFace dataset {ref.repo_id!r} loaded at an unpinned revision "
            f"({ref.revision or 'default branch'}); this is mutable and silently breaks "
            "reproducibility. Pin the run to the resolved commit: "
            f"https://huggingface.co/datasets/{ref.repo_id}/resolve/{pinned}/{filename}",
            stacklevel=2,
        )

    return ResolvedHfDataset(
        local_path=local_path,
        repo_id=ref.repo_id,
        revision=ref.revision,
        path=filename,
        commit_sha=commit_sha,
    )


def hf_dataset_provenance(dataset_ref) -> Optional[dict]:
    """Return a JSON-serializable provenance record for an HF ``dataset:`` value.

    ``None`` for local paths. For an HF URL, resolves it (cached) and reports
    the repo, the requested revision, and the exact commit SHA the run read --
    intended to be persisted into ``suite.json`` so a completed run records the
    precise dataset snapshot it used.
    """
    if not is_hf_ref(dataset_ref):
        return None
    resolved = resolve_hf_dataset(str(dataset_ref))
    return {
        "source": "huggingface",
        "url": str(dataset_ref),
        "repo_id": resolved.repo_id,
        "path": resolved.path,
        "revision": resolved.revision,
        "commit_sha": resolved.commit_sha,
    }
