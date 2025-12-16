import os
import sys
from pathlib import Path

from letta_client import AsyncLetta
from letta_client.types import CreateBlockParam

from letta_evals.decorators import agent_factory
from letta_evals.models import Sample

# add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from inventory_tool import TEST_TOOL_NAME


@agent_factory
async def create_inventory_agent(client: AsyncLetta, sample: Sample) -> str:
    """Create an inventory management agent using the Letta SDK.

    The agent is customized with item details from sample.agent_args.
    """
    tools_page = await client.tools.list(name=TEST_TOOL_NAME)
    if not tools_page.items:
        raise RuntimeError(f"Tool '{TEST_TOOL_NAME}' not found. Please ensure setup has been run.")
    tool = tools_page.items[0]

    item = sample.agent_args["item"]
    item_context = f"""Target Item Details:
- SKU: {item.get("sku", "Unknown")}
- Name: {item.get("name", "Unknown")}
- Price: ${item.get("price", 0.00)}
- Category: {item.get("category", "Unknown")}"""

    agent = await client.agents.create(
        name="inventory-assistant",
        memory_blocks=[
            CreateBlockParam(
                label="persona",
                value="You are a helpful inventory management assistant.",
            ),
            CreateBlockParam(
                label="item_context",
                value=item_context,
            ),
        ],
        agent_type="letta_v1_agent",
        model="openai/gpt-4.1-mini",
        embedding="openai/text-embedding-3-small",
        tool_ids=[tool.id],
        include_base_tools=False,
        project_id=os.environ.get("LETTA_PROJECT_ID"),
    )

    return agent.id
