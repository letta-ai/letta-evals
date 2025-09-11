from typing import Dict, Type

from letta_evals.graders.extractors.base import SubmissionExtractor
from letta_evals.graders.extractors.builtin import (
    AfterMarkerExtractor,
    AllAssistantExtractor,
    FirstAssistantExtractor,
    JSONExtractor,
    LastAssistantExtractor,
    LastTurnExtractor,
    PatternExtractor,
    ToolOutputExtractor,
)

EXTRACTOR_REGISTRY: Dict[str, Type[SubmissionExtractor]] = {
    "last_assistant": LastAssistantExtractor,
    "first_assistant": FirstAssistantExtractor,
    "all_assistant": AllAssistantExtractor,
    "last_turn": LastTurnExtractor,
    "pattern": PatternExtractor,
    "json": JSONExtractor,
    "tool_output": ToolOutputExtractor,
    "after_marker": AfterMarkerExtractor,
}


def register_extractor(name: str):
    """Decorator to register custom extractors."""

    def decorator(cls: Type[SubmissionExtractor]):
        EXTRACTOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_extractor(name: str, config: dict = None) -> SubmissionExtractor:
    """Get an extractor instance by name."""
    if name not in EXTRACTOR_REGISTRY:
        raise ValueError(f"Unknown extractor: {name}")

    extractor_class = EXTRACTOR_REGISTRY[name]
    return extractor_class(config=config)
