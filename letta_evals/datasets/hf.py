"""HuggingFace Hub-backed dataset resolution.

Resolves an HF dataset reference in a suite's ``dataset:`` field to a local
file, which the loader then reads exactly as it would a local path. Only the
single manifest file named in the reference is fetched -- never the whole repo
-- so large repos stay cheap, and on the Modal sandbox path nothing is fetched
in-sandbox (the host loads the dataset and ships each sample as JSON via
``--sample``).

``huggingface_hub`` is imported lazily -- only when an HF ref is actually used
-- and ships as an optional extra: ``pip install 'letta-evals[hf]'``. Suites
that only use local paths never pay for it.
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
    """Return True if ``value`` is a HuggingFace Hub URL.

    Only ``https://huggingface.co/...`` (and the http variant) count as HF
    references; everything else -- local path strings and ``Path`` objects --
    is left untouched so filesystem datasets behave exactly as before.
    """
    if not isinstance(value, str):
        return False
    v = value.strip().lower()
    return v.startswith(f"https://{_HF_HOST}/") or v.startswith(f"http://{_HF_HOST}/")


@dataclass(frozen=True)
class HfDatasetRef:
    """A parsed HuggingFace Hub reference to a single manifest file."""

    repo_id: str
    repo_type: str  # "dataset" | "model" | "space"
    revision: Optional[str]  # None => unpinned; resolves to the repo default branch
    path: Optional[str]  # None => bare repo URL; the manifest is resolved on fetch


@dataclass(frozen=True)
class ResolvedHfDataset:
    """The outcome of fetching an ``HfDatasetRef`` to local disk."""

    local_path: Path
    repo_id: str
    repo_type: str
    revision: Optional[str]
    path: str
    commit_sha: Optional[str]  # the exact commit the run read, when recoverable


def parse_hf_ref(value: str) -> HfDatasetRef:
    """Parse a huggingface.co URL into an :class:`HfDatasetRef`.

    Accepts the repo landing URL and the file (``/resolve/`` or ``/blob/``)
    URL::

        https://huggingface.co/datasets/<org>/<repo>
        https://huggingface.co/datasets/<org>/<repo>/resolve/<rev>/<path>
        https://huggingface.co/datasets/<org>/<repo>/blob/<rev>/<path>

    Model and Space repos are recognized via the leading path segment: no
    prefix => model, ``spaces/`` => space, ``datasets/`` => dataset.
    """
    parsed = urlparse(value.strip())
    segments = [unquote(s) for s in parsed.path.split("/") if s]

    if segments and segments[0] == "datasets":
        repo_type = "dataset"
        segments = segments[1:]
    elif segments and segments[0] == "spaces":
        repo_type = "space"
        segments = segments[1:]
    else:
        repo_type = "model"

    if len(segments) < 2:
        raise ValueError(
            f"Cannot parse HuggingFace dataset URL {value!r}: expected "
            "'https://huggingface.co/datasets/<org>/<repo>[/resolve/<rev>/<file>]'."
        )

    repo_id = f"{segments[0]}/{segments[1]}"
    rest = segments[2:]

    if not rest:
        # Bare repo URL: manifest file and revision are resolved on fetch.
        return HfDatasetRef(repo_id=repo_id, repo_type=repo_type, revision=None, path=None)

    if rest[0] in ("resolve", "blob"):
        if len(rest) < 3:
            raise ValueError(
                f"HuggingFace file URL {value!r} is missing a revision and/or file path; "
                "expected '.../resolve/<rev>/<file>'."
            )
        return HfDatasetRef(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=rest[1],
            path="/".join(rest[2:]),
        )

    raise ValueError(
        f"Unrecognized HuggingFace URL shape {value!r}; expected a repo URL or a "
        "'.../resolve/<rev>/<file>' file URL."
    )


def _is_unpinned(revision: Optional[str]) -> bool:
    """Whether ``revision`` is a mutable ref that breaks reproducibility.

    Telling a tag from a branch needs a network round-trip, so we only flag the
    unambiguous mutable cases: no revision, and the default branch names. A
    commit SHA or a tag is treated as pinned and does not warn.
    """
    return revision is None or revision in ("main", "master")


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


def _import_hf():
    try:
        import huggingface_hub
    except ImportError as e:
        raise ImportError(
            "HuggingFace-backed datasets require the 'huggingface_hub' package. "
            "Install it with: pip install 'letta-evals[hf]'."
        ) from e
    return huggingface_hub


def _resolve_manifest_filename(hub, ref: HfDatasetRef) -> str:
    """Find the single ``.jsonl``/``.csv`` manifest in a bare-repo reference."""
    files = hub.list_repo_files(repo_id=ref.repo_id, repo_type=ref.repo_type, revision=ref.revision)
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
    ref = parse_hf_ref(value)
    hub = _import_hf()

    filename = ref.path if ref.path is not None else _resolve_manifest_filename(hub, ref)

    local_path = Path(
        hub.hf_hub_download(
            repo_id=ref.repo_id,
            filename=filename,
            repo_type=ref.repo_type,
            revision=ref.revision,
        )
    )
    commit_sha = _commit_sha_from_cache_path(local_path)

    if _is_unpinned(ref.revision):
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
        repo_type=ref.repo_type,
        revision=ref.revision,
        path=filename,
        commit_sha=commit_sha,
    )
