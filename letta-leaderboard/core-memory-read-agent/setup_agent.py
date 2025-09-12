"""
Setup script to populate agent core memory with facts for each test case.
"""
import json
from typing import Dict, Any

from letta_client import AsyncLetta
from letta_evals.models import Sample


async def setup_core_memory(client: AsyncLetta, agent_id: str, sample: Sample) -> None:
    """
    Setup the agent's core memory with facts needed to answer the question.
    
    This function is called before each test case to populate the agent's
    core memory with the relevant facts from the sample metadata.
    """
    
    # Extract facts from sample metadata
    # Facts are stored in the extra field of SampleMetadata
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
        # Get current agent state to see existing memory blocks
        agent_state = await client.agents.retrieve(agent_id)
        
        # Find a memory block we can use for supporting facts
        # Look for existing blocks or create content for any available block
        memory_blocks = agent_state.memory.blocks if agent_state.memory else []
        
        if not memory_blocks:
            print(f"No memory blocks found for agent {agent_id}")
            return
        
        # Use the first memory block (usually there's a persona or similar block)
        target_block = memory_blocks[0]
        
        print(f"Found memory block: {target_block.label}")
        print(f"Setting up facts for question: {sample.input}")
        print(f"Expected answer: {sample.ground_truth}")
        
        # Update the memory block with supporting facts
        # Use the blocks.modify method to update the block
        await client.agents.blocks.modify(
            agent_id=agent_id,
            block_label=target_block.label,
            label="Supporting Facts",
            value=facts_context
        )
        
        print(f"✓ Updated memory block '{target_block.label}' with {len(facts)} supporting facts")
        
        # Verify the update worked
        updated_agent = await client.agents.retrieve(agent_id)
        updated_block = None
        if updated_agent.memory and updated_agent.memory.blocks:
            for block in updated_agent.memory.blocks:
                if block.id == target_block.id:
                    updated_block = block
                    break
        
        if updated_block:
            print(f"✓ Memory block now contains: {updated_block.value[:200]}...")
        
    except Exception as e:
        print(f"✗ Error updating core memory for agent {agent_id}: {e}")
        import traceback
        traceback.print_exc()
        raise