#!/usr/bin/env python3
"""Dev server that rebuilds data files before serving /data/ requests."""

import http.server
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIDECAR_ROOT = ROOT.parent

_last_build_mtime = 0.0
_build_lock = threading.Lock()


def _max_source_mtime() -> float:
    """Get the max mtime across all source files that feed build_data.py."""
    best = 0.0
    for category_dir in SIDECAR_ROOT.iterdir():
        if not category_dir.is_dir() or category_dir.name == "viewer":
            continue
        data_dir = category_dir / "data"
        if not data_dir.exists():
            continue
        for pattern in (
            "data/*.jsonl",
            "results/**/*",
            "rubric_*.txt",
            "judge_prompt.md",
        ):
            for f in category_dir.glob(pattern):
                try:
                    if f.is_file():
                        best = max(best, f.stat().st_mtime)
                except OSError:
                    continue
    return best


def _rebuild_if_needed():
    global _last_build_mtime
    with _build_lock:
        current = _max_source_mtime()
        if current <= _last_build_mtime:
            return
        result = subprocess.run(
            [sys.executable, str(ROOT / "build_data.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            _last_build_mtime = current
            print(f"  [rebuild] OK")
        else:
            print(f"  [rebuild] FAILED (exit {result.returncode})")
            if result.stderr:
                print(f"  {result.stderr.strip()}")


class RebuildHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path.startswith("/data/"):
            _rebuild_if_needed()
        super().do_GET()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5173
    server = http.server.HTTPServer(("", port), RebuildHandler)
    print(f"\n  Dev server running (auto-rebuilds on /data/ requests)\n")
    print(f"  -> http://localhost:{port}/\n")
    server.serve_forever()
