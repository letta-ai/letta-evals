"""Runner base URL resolution tests."""

from pathlib import Path

from letta_evals.decorators import extractor, grader
from letta_evals.models import GradeResult, SuiteSpec
from letta_evals.runner import Runner


@extractor
def _dummy_extractor(trajectory, config) -> str:
    return ""


@grader
def _dummy_grader(sample, submission) -> GradeResult:
    return GradeResult(score=1.0)


def _suite_without_target_base_url() -> SuiteSpec:
    this_file = Path(__file__).resolve().as_posix()
    return SuiteSpec.from_yaml(
        {
            "name": "base-url-resolution",
            "dataset": "samples.jsonl",
            "target": {
                "kind": "letta_code",
                "model_handles": ["letta/auto-fast"],
            },
            "graders": {
                "score": {
                    "kind": "tool",
                    "function": f"{this_file}:_dummy_grader",
                    "extractor": f"{this_file}:_dummy_extractor",
                }
            },
            "reward": {"kind": "metric", "metric_key": "score"},
        }
    )


def test_env_base_url_reaches_letta_code_target(monkeypatch):
    monkeypatch.setenv("LETTA_BASE_URL", "https://rollouts.example.test")
    suite = _suite_without_target_base_url()

    runner = Runner(suite=suite, max_concurrent=1)

    target = runner._create_letta_code_target("letta/auto-fast")
    assert target.base_url == "https://rollouts.example.test"


def test_cli_base_url_override_reaches_letta_code_target(monkeypatch):
    monkeypatch.setenv("LETTA_BASE_URL", "https://env.example.test")
    suite = _suite_without_target_base_url()

    runner = Runner(suite=suite, max_concurrent=1, letta_base_url="https://cli.example.test")

    target = runner._create_letta_code_target("letta/auto-fast")
    assert target.base_url == "https://cli.example.test"
