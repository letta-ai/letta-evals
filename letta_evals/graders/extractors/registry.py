import inspect
from pathlib import Path
from typing import Dict, Type

from letta_evals.graders.extractors.base import SubmissionExtractor
from letta_evals.graders.extractors.builtin import (
    AfterMarkerExtractor,
    AllAssistantExtractor,
    FirstAssistantExtractor,
    LastAssistantExtractor,
    LastTurnExtractor,
    PatternExtractor,
    ToolArgumentsExtractor,
    ToolOutputExtractor,
)
from letta_evals.utils.module_loader import load_object

EXTRACTOR_REGISTRY: Dict[str, Type[SubmissionExtractor]] = {
    "last_assistant": LastAssistantExtractor,
    "first_assistant": FirstAssistantExtractor,
    "all_assistant": AllAssistantExtractor,
    "last_turn": LastTurnExtractor,
    "pattern": PatternExtractor,
    "tool_arguments": ToolArgumentsExtractor,
    "tool_output": ToolOutputExtractor,
    "after_marker": AfterMarkerExtractor,
}


def register_extractor(name: str):
    """Decorator to register custom extractors."""

    def decorator(cls: Type[SubmissionExtractor]):
        EXTRACTOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_extractor(name: str, config: dict = None, base_dir: Path = None) -> SubmissionExtractor:
    """Get an extractor instance by name or file path."""
    # try registry first
    if name in EXTRACTOR_REGISTRY:
        extractor_class = EXTRACTOR_REGISTRY[name]
        return extractor_class(config=config)

    # try loading from file path
    if ":" in name:
        obj = load_object(name, base_dir=base_dir)
        # if it's a class, instantiate it
        if inspect.isclass(obj) and issubclass(obj, SubmissionExtractor):
            return obj(config=config)
        # if it's a function, wrap it in a simple extractor
        elif callable(obj):

            class FunctionExtractor(SubmissionExtractor):
                def extract(self, trajectory):
                    return obj(trajectory)

            return FunctionExtractor(config=config)
        else:
            raise ValueError(f"Loaded object {name} is not a valid extractor")

    raise ValueError(f"Unknown extractor: {name}")
