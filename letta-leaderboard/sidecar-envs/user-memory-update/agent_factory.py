"""Agent factory for user-memory-update eval.

Creates agents with per-sample initial memory from extra_vars.
"""

from letta_client import AsyncLetta
from letta_client.types import CreateBlockParam

from letta_evals.decorators import agent_factory
from letta_evals.models import Sample


@agent_factory
async def create_user_memory_agent(client: AsyncLetta, sample: Sample) -> str:
    """Create an agent with pre-populated user memory.

    Reads initial_memory from sample.extra_vars and creates an agent
    with that content in the 'user' memory block.

    Uses Letta's default system prompt and embedding.
    """
    extra_vars = sample.extra_vars or {}
    initial_memory = extra_vars.get("initial_memory", "")

    if not initial_memory:
        raise ValueError(f"No initial_memory in sample extra_vars: {sample.id}")

    agent = await client.agents.create(
        name=f"user-memory-{sample.id}",
        memory_blocks=[
            CreateBlockParam(
                label="user",
                value=initial_memory,
                limit=5000,
            ),
        ],
        model="openai/gpt-4o-mini",  # Overridden by suite config
    )

    print(f"✓ Created agent {agent.id} with {len(initial_memory)} chars initial memory")
    return agent.id
