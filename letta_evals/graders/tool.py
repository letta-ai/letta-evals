import inspect
from pathlib import Path
from typing import List, Optional, Tuple

from letta_client import AgentState, LettaMessageUnion

from letta_evals.decorators import GRADER_REGISTRY
from letta_evals.extractors import extractor_requires_agent_state, get_extractor
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
        self.extractor_name = extractor
        self.base_dir = base_dir
        self.extractor = get_extractor(extractor, extractor_config, base_dir=base_dir)
        self._requires_agent_state = extractor_requires_agent_state(extractor, base_dir=base_dir)

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

    @property
    def requires_agent_state(self) -> bool:
        """Whether this grader's extractor requires agent_state."""
        return self._requires_agent_state

    async def grade(
        self, sample: Sample, trajectory: List[List[LettaMessageUnion]], agent_state: Optional[AgentState] = None
    ) -> Tuple[GradeResult, str]:
        """Grade using the tool function."""
        submission = self.extractor(trajectory, agent_state=agent_state)

        # check if grader function is async
        if inspect.iscoroutinefunction(self.func):
            result = await self.func(sample, submission)
        else:
            result = self.func(sample, submission)

        return result, submission
