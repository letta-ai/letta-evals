#!/usr/bin/env python3
"""
Test script to manually verify memory setup functionality.
"""
import asyncio
import json
from letta_client import AsyncLetta
from setup_agent import setup_core_memory
from letta_evals.models import Sample, SampleMetadata

async def test_memory_setup():
    """Test the memory setup functionality manually."""
    
    # Load a sample from the dataset
    with open("datasets/core_memory_read.jsonl", 'r') as f:
        line = f.readline()
        sample_data = json.loads(line)
    
    # Convert to Sample object with proper metadata structure
    sample = Sample(
        input=sample_data["input"],
        ground_truth=sample_data["ground_truth"],
        metadata=SampleMetadata(
            tags=sample_data["metadata"].get("tags", []),
            extra=sample_data["metadata"].get("extra", {})
        )
    )
    
    print(f"Sample question: {sample.input}")
    print(f"Expected answer: {sample.ground_truth}")
    print(f"Facts available: {len(sample.metadata.extra.get('facts', [])) if sample.metadata and sample.metadata.extra else 'No facts'}")
    
    # Connect to Letta
    client = AsyncLetta(base_url="http://localhost:8283")
    
    # Import agent
    with open("core-memory-read-agent-v2.af", "rb") as f:
        resp = await client.agents.import_file(
            file=f, append_copy_suffix=False, override_existing_tools=False
        )
        agent_id = resp.agent_ids[0]
    
    print(f"Agent imported with ID: {agent_id}")
    
    # Test memory setup
    try:
        await setup_core_memory(client, agent_id, sample)
        print("✓ Memory setup completed successfully")
    except Exception as e:
        print(f"✗ Memory setup failed: {e}")
        return
    
    # Test asking the question
    from letta_client import MessageCreate
    
    letta_resp = await client.agents.messages.create(
        agent_id=agent_id,
        messages=[MessageCreate(role="user", content=sample.input)],
    )
    
    # Find the assistant response
    assistant_response = None
    print(f"Response messages: {len(letta_resp.messages)}")
    for i, msg in enumerate(letta_resp.messages):
        print(f"Message {i}: type={type(msg).__name__}, attributes={[attr for attr in dir(msg) if not attr.startswith('_')]}")
        if hasattr(msg, 'message_type') and msg.message_type == "assistant_message" and hasattr(msg, 'content') and msg.content:
            assistant_response = msg.content
            print(f"Found assistant message: {assistant_response}")
            break
    
    print(f"\nAgent response: {assistant_response}")
    print(f"Expected: {sample.ground_truth}")
    
    # Check if answer is in response
    if sample.ground_truth.lower() in assistant_response.lower():
        print("✓ SUCCESS: Expected answer found in response!")
    else:
        print("✗ FAILED: Expected answer not found in response")
    
    # Clean up
    await client.agents.delete(agent_id)
    print(f"Cleaned up agent {agent_id}")

if __name__ == "__main__":
    asyncio.run(test_memory_setup())