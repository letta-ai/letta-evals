import json
import re
import asyncio
from pathlib import Path

from letta_evals.decorators import grader
from letta_evals.models import GradeResult, Sample


def extract_json_from_submission(submission: str) -> dict:
    """Extract the file path from the submission."""
    try:
        # Try to find JSON within ```json ... ``` or ``` ... ``` code blocks
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)```', submission, re.DOTALL)
        if json_match:
            submission = json_match.group(1).strip()
        else:
            # Try to find a JSON object in the submission (look for { ... })
            # Find the last occurrence of a complete JSON object
            json_obj_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}(?:\s*)$', submission, re.DOTALL)
            if json_obj_match:
                submission = json_obj_match.group(0).strip()
            else:
                # If no JSON object pattern found, try to parse the whole submission
                submission = submission.strip()
        
        submission = json.loads(submission)
        file_path = submission.get("file_path")
        if not file_path:
            raise ValueError("No file_path found in JSON")
        return file_path
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nSubmission: {submission}")


@grader
async def python_output_grader(sample: Sample, submission: str) -> GradeResult:
    """Run the Python file and compare output to ground truth."""
    print(f"Submission: {submission}")

    # extract file path from submission
    file_path = extract_json_from_submission(submission)
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        return GradeResult(score=0.0, rationale=f"File not found: {file_path}")

    # run the python file
    try:
        process = await asyncio.create_subprocess_exec(
            "python3",
            str(file_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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
