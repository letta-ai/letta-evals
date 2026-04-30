"""Agent factory for the prose-memfs filesystem eval.

For each sample, create a fresh Letta agent and seed it with the rendered
prose corpus:

- The single ``system/index.md`` is registered as an in-context memory block
  (label ``system/index``) so it auto-projects into the system prompt.
- The bulky ``reference/`` tree (~510 files) is **not pushed to the server**.
  Instead we clone the agent's empty server-side memfs repo into the local
  cache at ``~/.letta/agents/$agent_id/memory/``, copy the rendered corpus
  in, and commit locally — no ``git push``. letta-code's subprocess runs
  with ``cwd`` set to that path (via ``permission_mode: memory``) and uses
  the seeded local state directly.

This trades server load for per-sample local compute. Each sample touches
~510 files inside a single git transaction; should be sub-second on SSD.

Requires ``LETTA_API_KEY`` to clone the (empty) server repo. Without that
clone we'd start from a fresh ``git init`` whose remote/HEAD wouldn't match
what letta-code expects, so the clone is the safer setup.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from letta_client import AsyncLetta

from letta_evals.decorators import agent_factory
from letta_evals.models import Sample


HERE = Path(__file__).parent
MEMFS_SRC = HERE / "memfs"


# ---------- frontmatter parsing ----------


def _split_frontmatter(text):
    """Parse a leading YAML-ish frontmatter block (only flat string keys).

    Returns ({key: value}, body). If no frontmatter is present, returns ({}, text).
    """
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    block = text[4:end]
    body = text[end + 5 :]
    front = {}
    for line in block.splitlines():
        line = line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if ":" in line:
            k, _, v = line.partition(":")
            front[k.strip()] = v.strip().strip('"').strip("'")
    return front, body


# ---------- local memfs seeding ----------


def _resolve_clone_url(agent_id: str) -> str:
    api_key = os.environ.get("LETTA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LETTA_API_KEY not set — required to clone the agent's "
            "server-side memfs repo for local seeding"
        )
    base_url = os.environ.get("LETTA_BASE_URL", "https://api.letta.com").rstrip("/")
    if not base_url.startswith("https://"):
        raise RuntimeError(
            f"LETTA_BASE_URL must be https for git clone auth injection; got {base_url!r}"
        )
    return f"https://letta:{api_key}@{base_url[len('https://'):]}/v1/git/{agent_id}/state.git"


def _seed_local_memfs(agent_id: str, memfs_src: Path) -> Path:
    """Populate the local memory clone for `agent_id`. Commit locally, no push.

    Returns the path of the populated repo.
    """
    local_dir = Path.home() / ".letta" / "agents" / agent_id / "memory"

    # If a previous run left state behind, wipe it. We always clone fresh so
    # the remote/HEAD config matches what letta-code expects to see.
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    auth_url = _resolve_clone_url(agent_id)

    result = subprocess.run(
        ["git", "clone", "--quiet", auth_url, str(local_dir)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git clone failed for agent {agent_id}: {result.stderr.strip()}"
        )

    # Copy rendered reference/* into the cloned repo. We deliberately skip
    # system/* — those go through the SDK as memory blocks, not via git.
    src_ref = memfs_src / "reference"
    dst_ref = local_dir / "reference"
    if dst_ref.exists():
        shutil.rmtree(dst_ref)
    shutil.copytree(src_ref, dst_ref)

    subprocess.run(
        ["git", "-C", str(local_dir), "add", "-A"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Skip the commit if there's nothing to add (paranoia: empty corpus).
    diff_check = subprocess.run(
        ["git", "-C", str(local_dir), "diff", "--cached", "--quiet"],
        capture_output=True,
    )
    if diff_check.returncode == 0:
        return local_dir

    subprocess.run(
        [
            "git", "-C", str(local_dir),
            "-c", "user.email=filesystem-memfs@letta.com",
            "-c", "user.name=filesystem-memfs-seed",
            "commit", "-m", "seed: prose memory corpus (local-only, not pushed)",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Intentionally **do not** push.
    return local_dir


# ---------- system block ----------


def _build_system_blocks():
    """Read system/index.md from the rendered corpus and return SDK block payloads."""
    path = MEMFS_SRC / "system" / "index.md"
    if not path.exists():
        raise RuntimeError(
            f"Rendered corpus not found at {MEMFS_SRC}. Run render_memfs.py first."
        )
    front, body = _split_frontmatter(path.read_text(encoding="utf-8"))
    description = front.get("description", "Memory navigation index")
    value = body.lstrip("\n")
    return [
        {
            "label": "system/index",  # letta-code appends .md when rendering
            "description": description,
            "value": value,
        }
    ]


# ---------- factory ----------


@agent_factory
async def setup_memfs_agent(client: AsyncLetta, sample: Sample) -> str:
    """Create a per-sample agent seeded with the prose corpus.

    The seeding has two halves:
      1. ``system/index`` block — created via SDK, lives server-side, gets
         projected into every system prompt automatically.
      2. ``reference/*`` files — written into the local memfs clone and
         committed (not pushed). letta-code's subprocess uses the local state
         directly via ``permission_mode: memory``.
    """
    blocks = _build_system_blocks()

    agent = await client.agents.create(
        name=f"filesystem-memfs-{sample.id}",
        memory_blocks=blocks,
    )

    _seed_local_memfs(agent.id, MEMFS_SRC)

    return agent.id
