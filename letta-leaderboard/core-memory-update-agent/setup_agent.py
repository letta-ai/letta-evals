"""
Setup script to create agent with core memory populated with facts and then updated with contradicting facts.
"""

from letta_client import AsyncLetta, CreateBlock

from letta_evals.decorators import agent_factory
from letta_evals.models import Sample


@agent_factory
async def setup_agent(client: AsyncLetta, sample: Sample) -> str:
    """
    Create an agent for core memory update tasks.

    This function creates an agent with initial facts in core memory.
    The contradicting fact will be sent during the evaluation process.

    Args:
        client: The AsyncLetta client
        sample: The sample containing facts

    Returns:
        str: The ID of the created agent
    """

    # Extract facts from sample agent_args
    facts = []

    if sample.agent_args and "extra" in sample.agent_args and sample.agent_args["extra"]:
        facts = sample.agent_args["extra"].get("facts", [])

    if not facts:
        raise ValueError(f"No facts available for sample: {sample.input}")

    # Format facts as numbered list for core memory
    facts_context = "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(facts))

    try:
        # Create agent with facts pre-populated in core memory
        agent = await client.agents.create(
            name="Core Memory Updater",
            memory_blocks=[
                CreateBlock(
                    label="persona",
                    value="You are an AI assistant that answers questions based on the facts stored in your core memory. When you receive new information that contradicts existing facts, you should update your Supporting Facts memory block to reflect the new information. Always use the most recent information to answer questions accurately.",
                ),
                CreateBlock(label="Supporting Facts", value=facts_context),
            ],
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
            agent_type="letta_v1_agent",
        )

        print(f"✓ Created agent {agent.id} with {len(facts)} initial supporting facts")

        return agent.id

    except Exception as e:
        print(f"✗ Error creating agent: {e}")
        import traceback

        traceback.print_exc()
        raise
