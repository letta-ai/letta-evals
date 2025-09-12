from pathlib import Path
from typing import List, Optional

from letta_client import LettaMessageUnion

from letta_evals.decorators import GRADER_REGISTRY
from letta_evals.extractors import get_extractor
from letta_evals.graders.base import Grader
from letta_evals.models import GradeResult, Sample
from letta_evals.utils import load_object


class ToolGrader(Grader):
    """Grader that uses Python functions."""

    def __init__(
        self,
        function: str,
        extractor: str = "last_assistant",
        extractor_config: Optional[dict] = None,
        base_dir: Optional[Path] = None,
    ):
        self.function_name = function
        self.extractor = get_extractor(extractor, extractor_config, base_dir=base_dir)

        if function in GRADER_REGISTRY:
            self.func = GRADER_REGISTRY[function]
        elif ":" in function:
            obj = load_object(function, base_dir=base_dir)
            if callable(obj) and hasattr(obj, "_is_grader"):
                self.func = obj
            else:
                raise ValueError(
                    f"Loaded object {function} is not a valid @grader decorated function. "
                    f"Please use the @grader decorator."
                )
        else:
            raise ValueError(f"Grader function '{function}' not found in registry")

    async def grade(self, sample: Sample, trajectory: List[List[LettaMessageUnion]]) -> GradeResult:
        """Grade using the tool function."""
        submission = self.extractor(trajectory)
        return self.func(sample, submission)
