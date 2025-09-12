from abc import ABC, abstractmethod

from letta_client import AsyncLetta


class AgentFactory(ABC):
    """Base interface for programmatic agent creation."""

    @abstractmethod
    async def create(self, client: AsyncLetta) -> str:
        """Create and return agent ID using Letta SDK.

        Args:
            client: Authenticated Letta client to use for agent creation

        Returns:
            str: ID of the created agent
        """
        pass
