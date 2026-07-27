from __future__ import annotations

import json
from pathlib import Path

import anyio

from letta_evals.runner import run_suite


def _write_suite(tmp_path: Path, model_handles: list[str] | None) -> Path:
    dataset_path = tmp_path / "samples.jsonl"
    dataset_path.write_text(json.dumps({"id": "sample-1", "input": "hello"}) + "\n")

    model_yaml = ""
    if model_handles is not None:
        model_yaml = "  model_handles:\n" + "".join(f"    - {handle}\n" for handle in model_handles)

    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        f"name: model-override-test\n"
        f"dataset: samples.jsonl\n"
        f"target:\n"
        f"  kind: letta_code\n"
        f"{model_yaml}"
        f"graders:\n"
        f"  acc:\n"
        f"    kind: tool\n"
        f"    function: exact_match\n"
        f"reward:\n"
        f"  kind: metric\n"
        f"  metric_key: acc\n"
    )
    return suite_path


def test_run_suite_overrides_declared_model_handles(tmp_path, monkeypatch):
    suite_path = _write_suite(tmp_path, ["openai/model-a", "openai/model-b"])
    captured = {}
    sentinel = object()

    async def fake_execute_runs(**kwargs):
        captured["suite"] = kwargs["suite"]
        return sentinel

    monkeypatch.setattr("letta_evals.runner._execute_runs", fake_execute_runs)

    async def run():
        return await run_suite(
            suite_path,
            max_concurrent=1,
            model_handle="baseten/dream-1@letta-research",
        )

    result = anyio.run(run)

    assert result is sentinel
    assert captured["suite"].target.model_handles == ["baseten/dream-1@letta-research"]


def test_run_suite_override_sets_missing_model_handles(tmp_path, monkeypatch):
    suite_path = _write_suite(tmp_path, None)
    captured = {}

    async def fake_execute_runs(**kwargs):
        captured["suite"] = kwargs["suite"]
        return object()

    monkeypatch.setattr("letta_evals.runner._execute_runs", fake_execute_runs)

    async def run():
        await run_suite(
            suite_path,
            max_concurrent=1,
            model_handle="baseten/dream-1@production",
        )

    anyio.run(run)

    assert captured["suite"].target.model_handles == ["baseten/dream-1@production"]


def test_run_suite_without_override_preserves_declared_models(tmp_path, monkeypatch):
    declared = ["openai/model-a", "openai/model-b"]
    suite_path = _write_suite(tmp_path, declared)
    captured = {}

    async def fake_execute_runs(**kwargs):
        captured["suite"] = kwargs["suite"]
        return object()

    monkeypatch.setattr("letta_evals.runner._execute_runs", fake_execute_runs)

    async def run():
        await run_suite(suite_path, max_concurrent=1)

    anyio.run(run)

    assert captured["suite"].target.model_handles == declared
