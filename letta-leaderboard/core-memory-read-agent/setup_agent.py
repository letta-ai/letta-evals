"""
Setup script to create agent with core memory populated with facts for each test case.
"""
import json
from typing import Dict, Any

from letta_client import AsyncLetta, CreateBlock
from letta_evals.models import Sample
from letta_evals.decorators import agent_factory


@agent_factory
async def setup_agent(client: AsyncLetta) -> str:
    """
    Create a basic agent for core memory reading tasks.

    This function creates a fresh agent that can be populated with facts
    during the evaluation process.

    Returns:
        str: The ID of the created agent
    """

    try:
        # Create a basic agent with instructions to use core memory
        agent = await client.agents.create(
            name="Core Memory Reader",
            memory_blocks=[
                CreateBlock(
                    label="persona",
                    value="You are an AI assistant that answers questions based on the facts stored in your core memory. Use only the information provided in your Supporting Facts memory block to answer questions accurately."
                ),
                CreateBlock(
                    label="Supporting Facts",
                    value="This block will be populated with relevant facts for each question."
                )
            ],
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small"
        )

        print(f"✓ Created basic agent {agent.id} for core memory reading")

        return agent.id

    except Exception as e:
        print(f"✗ Error creating agent: {e}")
        import traceback
        traceback.print_exc()
        raise


async def setup_core_memory(client: AsyncLetta, agent_id: str, sample: Sample) -> None:
    """
    Setup the agent's core memory with facts needed to answer the question.

    This function is called before each test case to populate the agent's
    core memory with the relevant facts from the sample metadata.
    """

    # Extract facts from sample metadata
    # Facts are stored in the extra field of SampleMetadata
    print(sample.metadata)
    facts = []
    if sample.metadata and hasattr(sample.metadata, 'extra'):
        facts = sample.metadata.extra.get('facts', [])

    print(f"[SETUP] Question: {sample.input[:100]}...")
    print(f"[SETUP] Expected answer: {sample.ground_truth}")
    print(f"[SETUP] Facts found: {len(facts)}")

    if facts:
        print(f"[SETUP] First fact: {facts[0][:100]}...")
        # Check if the expected answer appears in the facts
        facts_text = " ".join(facts).lower()
        if sample.ground_truth.lower() in facts_text:
            print(f"[SETUP] ✓ Expected answer found in facts")
        else:
            print(f"[SETUP] ✗ Expected answer NOT found in facts")
    else:
        print(f"[SETUP] ✗ No facts available")

    if not facts:
        raise ValueError(f"No facts available for sample: {sample.input}")

    # Format facts as numbered list
    facts_context = "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(facts))

    try:
        # Update the Supporting Facts memory block with the facts
        await client.agents.blocks.modify(
            agent_id=agent_id,
            block_label="Supporting Facts",
            label="Supporting Facts",
            value=facts_context
        )

        print(f"✓ Updated memory block with {len(facts)} supporting facts")

    except Exception as e:
        print(f"✗ Error updating core memory for agent {agent_id}: {e}")
        import traceback
        traceback.print_exc()
        raise