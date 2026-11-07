"""Scorer for the skill-use judge."""

from letta_evals.graders.scorers import ChecklistResponse, checklist

__all__ = ["ChecklistResponse", "skill_use"]


def skill_use(sample, response: ChecklistResponse) -> float:
    """Checklist score over rubric_skill_use, or 0 when the right skill wasn't selected.

    Criterion 1 of every rubric_skill_use is "the <skill> skill is selected".
    """
    score = checklist(sample, response, criteria_var="rubric_skill_use")
    first = sample.rubric_vars["rubric_skill_use"][0]["id"]
    if any(v.id == first and v.verdict == "UNMET" for v in response.criteria):
        return 0.0
    return score
