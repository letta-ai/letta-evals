from abc import ABC, abstractmethod
from typing import List, Optional

from letta_client import LettaMessageUnion


class SubmissionExtractor(ABC):
    """Base interface for extracting submissions from trajectories."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize extractor with optional configuration."""
        self.config = config or {}

    @abstractmethod
    def extract(self, trajectory: List[List[LettaMessageUnion]]) -> str:
        """Extract submission text from trajectory."""
        pass
