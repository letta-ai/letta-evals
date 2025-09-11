import importlib
from typing import Dict, List, Optional

from letta_client import LettaMessageUnion

from letta_evals.graders.base import Grader
from letta_evals.graders.extractors.registry import get_extractor
from letta_evals.models import GradeResult, Sample

GRADER_REGISTRY: Dict[str, callable] = {}


def register_grader(name: str):
    """Decorator to register a grader function."""

    def decorator(func):
        GRADER_REGISTRY[name] = func
        return func

    return decorator


@register_grader("exact_match")
def exact_match(sample: Sample, submission: str) -> GradeResult:
    """Check if submission exactly matches ideal."""
    if not sample.ideal:
        return GradeResult(score=0.0, rationale="No ideal answer provided")

    matches = submission.strip() == sample.ideal.strip()
    score = 1.0 if matches else 0.0
    return GradeResult(score=score, rationale=f"Exact match: {matches}")


@register_grader("contains")
def contains(sample: Sample, submission: str) -> GradeResult:
    """Check if submission contains ideal answer."""
    if not sample.ideal:
        return GradeResult(score=0.0, rationale="No ideal answer provided")

    found = sample.ideal.lower() in submission.lower()
    score = 1.0 if found else 0.0
    return GradeResult(score=score, rationale=f"Contains ideal: {found}")


class ToolGrader(Grader):
    """Grader that uses Python functions."""

    def __init__(
        self,
        function: str,
        module: Optional[str] = None,
        extractor: str = "last_assistant",
        extractor_config: Optional[dict] = None,
    ):
        self.function_name = function
        self.module = module
        self.extractor = get_extractor(extractor, extractor_config)

        if function in GRADER_REGISTRY:
            self.func = GRADER_REGISTRY[function]
        elif module:
            mod = importlib.import_module(module)
            self.func = getattr(mod, function)
        else:
            raise ValueError(f"Grader function '{function}' not found in registry and no module specified")

    async def grade(self, sample: Sample, trajectory: List[List[LettaMessageUnion]]) -> GradeResult:
        """Grade using the tool function."""
        submission = self.extractor.extract(trajectory)
        return self.func(sample, submission)
