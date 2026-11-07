"""Built-in response formats and scorers for model judges.

A model judge replies with a Pydantic model (its response format), and a
scorer turns that reply into a score. A scorer takes ``(sample, response)``
and returns a float in [0, 1]; raising ``ValueError`` rejects the reply, and
the judge is asked again.

Built-ins:
    score      the judge replies ``{rationale, score}`` and the score is used as is.
    checklist  the judge gives a MET/UNMET verdict on each criterion in
               ``sample.rubric_vars["criteria"]``, and the score is computed from them.
"""

from typing import Any, Dict, List, Literal, Set

from pydantic import BaseModel, ConfigDict, Field

from letta_evals.models import Sample

# ── score ────────────────────────────────────────────────────────────────────


class ScoreResponse(BaseModel):
    """A single holistic score. ``rationale`` comes first so the judge reasons before it scores."""

    rationale: str = Field(description="Explanation of the grading decision")
    score: float = Field(ge=0.0, le=1.0, description="Score between 0.0 and 1.0")


def score(sample: Sample, response: Any) -> float:
    return response.score


# ── checklist ────────────────────────────────────────────────────────────────


class Criterion(BaseModel):
    """One checklist criterion, as written in ``sample.rubric_vars["criteria"]``.

    A negative weight marks a criterion that describes an error: when the judge
    finds it MET, its weight is subtracted.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    weight: float = 1.0


class CriterionVerdict(BaseModel):
    id: str = Field(description="The criterion's id, exactly as given")
    rationale: str = Field(description="Evidence for the verdict")
    verdict: Literal["MET", "UNMET"]


class ChecklistResponse(BaseModel):
    criteria: List[CriterionVerdict] = Field(description="One verdict per criterion, each criterion exactly once")


def sample_criteria(sample: Sample) -> List[Criterion]:
    """Parse and check ``sample.rubric_vars["criteria"]``.

    Raises:
        ValueError: If the criteria are missing, malformed, have duplicate ids,
            or have no positive weight.
    """
    raw = (sample.rubric_vars or {}).get("criteria")
    if not isinstance(raw, list) or not raw:
        raise ValueError("scorer 'checklist' needs rubric_vars['criteria']: a non-empty list of {id, text, weight}")
    criteria = [Criterion.model_validate(item) for item in raw]
    ids = [c.id for c in criteria]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"criteria ids must be unique, duplicated: {duplicates}")
    if not any(c.weight > 0 for c in criteria):
        raise ValueError("criteria need at least one positive weight")
    return criteria


def checklist(sample: Sample, response: ChecklistResponse) -> float:
    """Weighted share of positive criteria MET, minus negative criteria MET, clipped to [0, 1]."""
    criteria = sample_criteria(sample)
    judged = [v.id for v in response.criteria]
    expected = {c.id for c in criteria}
    problems = []
    if missing := sorted(expected - set(judged)):
        problems.append(f"missing {missing}")
    if unknown := sorted(set(judged) - expected):
        problems.append(f"unknown {unknown}")
    if duplicated := sorted({i for i in judged if judged.count(i) > 1}):
        problems.append(f"duplicated {duplicated}")
    if problems:
        raise ValueError(f"each criterion must be judged exactly once: {'; '.join(problems)}")

    met = {v.id for v in response.criteria if v.verdict == "MET"}
    positive = sum(c.weight for c in criteria if c.weight > 0)
    earned = sum(c.weight for c in criteria if c.id in met)
    return min(1.0, max(0.0, earned / positive))


# ── response format checks ───────────────────────────────────────────────────


def check_response_model(model: Any) -> None:
    """Reject response formats that one of the providers can't serve.

    Anthropic turns a free-form object (``dict[str, X]``) into an empty object
    the judge can only fill with ``{}``, and OpenAI's strict mode rejects it.
    Anthropic rejects recursive models.

    Raises:
        TypeError: If ``model`` is not a Pydantic model class.
        ValueError: If the model has a free-form object or is recursive.
    """
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise TypeError(f"response_format must be a Pydantic model class, got {model!r}")

    schema = model.model_json_schema()
    defs: Dict[str, Any] = schema.get("$defs", {})

    def walk(node: Any, path: str, refs: Set[str]) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and (
                "properties" not in node or node.get("additionalProperties") not in (None, False)
            ):
                raise ValueError(
                    f"response_format {model.__name__} has a free-form object at {path or 'the top level'}; "
                    "use a list of models instead of a dict"
                )
            if isinstance(node.get("$ref"), str):
                refs.add(node["$ref"].rsplit("/", 1)[-1])
            for key, value in node.items():
                if key != "$defs":
                    walk(value, f"{path}.{key}" if path else key, refs)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]", refs)

    walk(schema, "", set())
    edges: Dict[str, Set[str]] = {}
    for name, definition in defs.items():
        edges[name] = set()
        walk(definition, name, edges[name])

    def reaches(start: str, target: str, seen: Set[str]) -> bool:
        for nxt in edges.get(start, ()):
            if nxt == target:
                return True
            if nxt not in seen:
                seen.add(nxt)
                if reaches(nxt, target, seen):
                    return True
        return False

    recursive = sorted(name for name in defs if reaches(name, name, set()))
    if recursive:
        raise ValueError(f"response_format {model.__name__} is recursive ({recursive}); Anthropic rejects that")
