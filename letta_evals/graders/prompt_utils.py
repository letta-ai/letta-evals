"""Prompt construction for rubric-based graders.

The rubric text is treated as a template that is substituted and sent
verbatim to the judge. The framework does not add wrapping text, headers,
or a system prompt of its own — those concerns belong to the rubric author.

Reserved template variables (always available):
    {input}         - the original input from sample.input
    {ground_truth}  - the expected answer (empty string if not provided)
    {submission}    - the agent's extracted output

Additional variables: any key in sample.rubric_vars is automatically
available as {key}. Reserved names cannot be reused in rubric_vars.

Missing placeholders raise KeyError so the failure is loud rather than
silently leaving literal "{var}" in the prompt sent to the judge.
"""

import string

from letta_evals.models import Sample

RESERVED_VARS = ("input", "ground_truth", "submission")


def build_judge_prompt(rubric: str, sample: Sample, submission: str) -> str:
    """Substitute reserved and rubric_vars placeholders into the rubric text.

    Args:
        rubric: The rubric template. Sent verbatim to the judge after substitution.
        sample: The evaluation sample (provides ``input``, ``ground_truth``,
            and ``rubric_vars``).
        submission: The extracted agent submission to evaluate.

    Returns:
        The substituted rubric text.

    Raises:
        ValueError: If ``sample.rubric_vars`` contains a reserved variable name.
        KeyError: If the rubric references a ``{placeholder}`` that is neither a
            reserved variable nor a key in ``sample.rubric_vars``.
    """
    substitutions = {
        "input": str(sample.input),
        "ground_truth": str(sample.ground_truth) if sample.ground_truth is not None else "",
        "submission": submission,
    }

    if sample.rubric_vars:
        for key, value in sample.rubric_vars.items():
            if key in RESERVED_VARS:
                raise ValueError(
                    f"rubric_vars key {key!r} collides with a reserved variable. Reserved names: {RESERVED_VARS}"
                )
            substitutions[key] = str(value)

    try:
        return string.Formatter().vformat(rubric, (), substitutions)
    except KeyError as e:
        missing = e.args[0] if e.args else "?"
        raise KeyError(
            f"Rubric references {{{missing}}} but no value was provided. "
            f"Available variables: {sorted(substitutions.keys())}. "
            f"Reserved variables ({list(RESERVED_VARS)}) come from the Sample; "
            f"add other variables via sample.rubric_vars."
        ) from e
