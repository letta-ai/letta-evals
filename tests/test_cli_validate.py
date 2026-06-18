from pathlib import Path

from typer.testing import CliRunner

from letta_evals.cli import app


def test_validate_command_accepts_letta_code_target(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"input": "hi", "ground_truth": "hi"}\n')
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "\n".join(
            [
                "name: cli-validate-test",
                "dataset: dataset.jsonl",
                "target:",
                "  kind: letta_code",
                "graders:",
                "  exact:",
                "    kind: tool",
                "    function: exact_match",
                "gate:",
                "  kind: simple",
                "  metric_key: exact",
                "  op: gte",
                "  value: 1.0",
                "",
            ]
        )
    )

    result = CliRunner().invoke(app, ["validate", str(suite)])

    assert result.exit_code == 0, result.output
    assert "Suite 'cli-validate-test' is valid" in result.output
    assert "Target: letta_code" in result.output
