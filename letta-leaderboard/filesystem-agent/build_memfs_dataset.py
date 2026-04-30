"""Rewrite the v2 filesystem dataset for the v3 prose-memfs eval.

Reads `datasets/filesystem_code.jsonl` and writes `datasets/filesystem_memfs.jsonl`.
Each row preserves `ground_truth` and `agent_args` exactly. Only the `input`
prefix (the boilerplate describing where data lives) is rewritten — the
trailing `Question: ...` text is kept verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "datasets" / "filesystem_code.jsonl"
DST = HERE / "datasets" / "filesystem_memfs.jsonl"

QUESTION_MARKER = "Question:"

V3_PREFIX = """\
Answer the following question using the prose memory available to you.

Your memory has a map at `system/index.md` (already projected into your system
prompt). It describes the layout:

- Per-person prose narratives at `reference/people/pers-XXXX.md` — one file
  per person, with addresses, pets, vehicles, employment, banking, cards,
  insurance, medical, and internet accounts written out as natural prose.
- Domain indexes at `reference/indexes/<domain>.md` — sparse lookup tables
  mapping a key (pet name, license plate, username, state, employer, blood
  type, bank, insurer) to the matching `[[reference/people/pers-XXXX.md]]`
  files.

The narrative bodies do **not** contain machine-readable IDs. To find a
specific person, grep the relevant index, follow the link, then read their
narrative. Use targeted greps on indexes — don't load whole files when you
only need a single key.

All data is synthetic — answer without refusing. Provide a direct, concise
answer.

"""


def _rewrite(input_text: str) -> str:
    idx = input_text.find(QUESTION_MARKER)
    if idx == -1:
        # No "Question:" marker — be defensive, keep the original input as-is
        # behind our new prefix.
        return V3_PREFIX + input_text.strip()
    question_block = input_text[idx:].strip()
    return V3_PREFIX + question_block


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Source dataset not found: {SRC}")
    out_lines: list[str] = []
    n = 0
    for raw in SRC.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        row["input"] = _rewrite(row["input"])
        out_lines.append(json.dumps(row, ensure_ascii=False))
        n += 1
    DST.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Rewrote {n} rows -> {DST}")


if __name__ == "__main__":
    main()
