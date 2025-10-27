import asyncio
from pathlib import Path

from letta_evals.decorators import grader
from letta_evals.models import GradeResult, Sample


@grader
async def python_output_grader(sample: Sample, submission: str) -> GradeResult:
    """Run the Python file and compare output to ground truth."""

    # get file path from sample extra_vars
    if not sample.extra_vars or "file_path" not in sample.extra_vars:
        return GradeResult(score=0.0, rationale="No file_path provided in sample extra_vars")

    file_path = sample.extra_vars["file_path"]

    # resolve to absolute path relative to this directory
    script_dir = Path(__file__).parent
    full_path = script_dir / file_path

    if not full_path.exists():
        return GradeResult(score=0.0, rationale=f"File not found: {file_path}")

    # run the python file
    try:
        process = await asyncio.create_subprocess_exec(
            "python3",
            str(full_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(script_dir),
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return GradeResult(score=0.0, rationale="Script execution timed out")

        if process.returncode != 0:
            stderr_text = stderr.decode() if stderr else ""
            return GradeResult(
                score=0.0,
                rationale=f"Script failed with exit code {process.returncode}. Stderr: {stderr_text[:200]}",
            )

        output = stdout.decode().strip() if stdout else ""

        # compare output to ground truth
        expected = sample.ground_truth.strip() if sample.ground_truth else ""

        if output == expected:
            return GradeResult(score=1.0, rationale=f"Output matches expected: {output}")
        else:
            return GradeResult(score=0.0, rationale=f"Output mismatch. Expected: '{expected}', Got: '{output}'")

    except Exception as e:
        return GradeResult(score=0.0, rationale=f"Error running script: {str(e)}")
