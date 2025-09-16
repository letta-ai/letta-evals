"""
Setup script to create agent with core memory populated with facts and then updated with contradicting facts.
"""
import json
from typing import Dict, Any

from letta_client import AsyncLetta, CreateBlock, MessageCreate
from letta_evals.models import Sample
from letta_evals.decorators import agent_factory


@agent_factory
async def setup_agent(client: AsyncLetta, sample: Sample) -> str:
    """
    Create an agent for core memory update tasks.

    This function creates an agent with initial facts in core memory,
    then sends a contradicting fact to update the memory before evaluation.

    Args:
        client: The AsyncLetta client
        sample: The sample containing facts and contradicting fact

    Returns:
        str: The ID of the created agent
    """

    # Extract facts and contradicting fact from sample agent_args
    facts = []
    contradicting_fact = None

    if sample.agent_args and 'extra' in sample.agent_args and sample.agent_args['extra']:
        facts = sample.agent_args['extra'].get('facts', [])
        contradicting_fact = sample.agent_args['extra'].get('contradicting_fact', None)

    if not facts:
        raise ValueError(f"No facts available for sample: {sample.input}")

    if not contradicting_fact:
        raise ValueError(f"No contradicting fact available for sample: {sample.input}")

    # Format facts as numbered list for core memory
    facts_context = "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(facts))

    try:
        # Create agent with facts pre-populated in core memory
        agent = await client.agents.create(
            name="Core Memory Updater",
            memory_blocks=[
                CreateBlock(
                    label="persona",
                    value="You are an AI assistant that answers questions based on the facts stored in your core memory. When you receive new information that contradicts existing facts, you should update your Supporting Facts memory block to reflect the new information. Always use the most recent information to answer questions accurately."
                ),
                CreateBlock(
                    label="Supporting Facts",
                    value=facts_context
                )
            ],
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small"
        )

        print(f"✓ Created agent {agent.id} with {len(facts)} initial supporting facts")

        # Send the contradicting fact to update the agent's memory
        await client.agents.messages.create(
            agent_id=agent.id,
            messages=[
                MessageCreate(
                    role="user",
                    content=f"Please update your knowledge with this new information: {contradicting_fact}"
                )
            ]
        )

        print(f"✓ Sent contradicting fact to agent for memory update")

        # Reset messages to clear the update interaction before evaluation
        await client.agents.messages.reset(agent_id=agent.id)
        print(f"✓ Reset agent messages for clean evaluation")

        return agent.id

    except Exception as e:
        print(f"✗ Error creating agent: {e}")
        import traceback
        traceback.print_exc()
        raise