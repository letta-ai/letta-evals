"""Smoke tests for keeping package imports lightweight."""

import os
import subprocess
import sys
from pathlib import Path


def test_package_import_does_not_eagerly_import_provider_sdks_or_pandas():
    code = """
import sys
import letta_evals

unexpected = [
    module_name
    for module_name in ("anthropic", "google.genai", "openai", "pandas")
    if module_name in sys.modules
]
if unexpected:
    raise AssertionError(f"unexpected eager imports: {unexpected}")
"""
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
