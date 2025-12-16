"""
Setup script to create agent with core memory populated with facts for each test case.
"""

from letta_client import AsyncLetta
from letta_client.types import CreateBlockParam

from letta_evals.decorators import agent_factory
from letta_evals.models import Sample


@agent_factory
async def setup_agent(client: AsyncLetta, sample: Sample) -> str:
    """
    Create a basic agent for core memory reading tasks.

    This function creates a fresh agent and populates it with facts
    from the sample for the evaluation.

    Args:
        client: The AsyncLetta client
        sample: The sample containing facts to populate core memory

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
    facts_context = "\n".join(f"{i + 1}. {fact}" for i, fact in enumerate(facts))

    try:
        # Create agent with facts pre-populated in core memory
        agent = await client.agents.create(
            name="Core Memory Reader",
            memory_blocks=[
                CreateBlockParam(
                    label="persona",
                    value="You are an AI assistant that answers questions based on the facts stored in your core memory. Use only the information provided in your Supporting Facts memory block to answer questions accurately.",
                ),
                CreateBlockParam(label="Supporting Facts", value=facts_context),
            ],
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
        )

        print(f"✓ Created agent {agent.id} with {len(facts)} supporting facts in core memory")

        return agent.id

    except Exception as e:
        print(f"✗ Error creating agent: {e}")
        import traceback

        traceback.print_exc()
        raise
