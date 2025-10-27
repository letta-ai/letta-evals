import subprocess
from pathlib import Path

from letta_evals.decorators import grader
from letta_evals.models import GradeResult, Sample


@grader
def python_output_grader(sample: Sample, submission: str) -> GradeResult:
    """Run the Python file and compare output to ground truth."""

    # get file path from sample rubric_vars
    if not sample.rubric_vars or "file_path" not in sample.rubric_vars:
        return GradeResult(score=0.0, rationale="No file_path provided in sample rubric_vars")

    file_path = sample.rubric_vars["file_path"]

    # resolve to absolute path relative to this directory
    script_dir = Path(__file__).parent
    full_path = script_dir / file_path

    if not full_path.exists():
        return GradeResult(score=0.0, rationale=f"File not found: {file_path}")

    # run the python file
    try:
        result = subprocess.run(
            ["python3", str(full_path)],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(script_dir),
        )

        if result.returncode != 0:
            return GradeResult(
                score=0.0, rationale=f"Script failed with exit code {result.returncode}. Stderr: {result.stderr[:200]}"
            )

        output = result.stdout.strip()

        # compare output to ground truth
        expected = sample.ground_truth.strip() if sample.ground_truth else ""

        if output == expected:
            return GradeResult(score=1.0, rationale=f"Output matches expected: {output}")
        else:
            return GradeResult(score=0.0, rationale=f"Output mismatch. Expected: '{expected}', Got: '{output}'")

    except subprocess.TimeoutExpired:
        return GradeResult(score=0.0, rationale="Script execution timed out")
    except Exception as e:
        return GradeResult(score=0.0, rationale=f"Error running script: {str(e)}")
