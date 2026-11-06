"""pytest configuration for letta-evals tests"""

import tempfile

import pytest


def pytest_addoption(parser):
    parser.addoption("--suite-path", action="store", default=None, help="Path to a specific suite.yaml file to test")


@pytest.fixture(autouse=True)
def _runner_temp_dirs_in_tmp_path(tmp_path, monkeypatch):
    """Create the runner's per-sample directories (working directories, local
    stores) under the test's tmp_path instead of leaving them in /tmp."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
