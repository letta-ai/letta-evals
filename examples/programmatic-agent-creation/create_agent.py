from typing import List, Type

from letta_client import AsyncLetta, CreateBlock
from letta_client.client import BaseTool
from pydantic import BaseModel

from letta_evals.decorators import agent_factory
from letta_evals.models import Sample


class InventoryItem(BaseModel):
    sku: str
    name: str
    price: float
    category: str


class InventoryEntry(BaseModel):
    timestamp: int
    item: InventoryItem
    transaction_id: str


class InventoryEntryData(BaseModel):
    data: InventoryEntry
    quantity_change: int


class ManageInventoryTool(BaseTool):
    name: str = "manage_inventory"
    args_schema: Type[BaseModel] = InventoryEntryData
    description: str = "Update inventory catalogue with a new data entry"
    tags: List[str] = ["inventory", "shop"]

    def run(self, data: InventoryEntry, quantity_change: int) -> str:
        return f"Updated inventory for {data.item.name} with a quantity change of {quantity_change}"


@agent_factory
async def create_inventory_agent(client: AsyncLetta, sample: Sample) -> str:
    """Create an inventory management agent using the Letta SDK.

    The agent is customized with item details from sample.agent_args.
    """
    tools = await client.tools.list(name="manage_inventory")
    if not tools:
        raise RuntimeError("Tool 'manage_inventory' not found. Please ensure setup has been run.")
    tool = tools[0]

    item = sample.agent_args["item"]
    item_context = f"""Target Item Details:
- SKU: {item.get("sku", "Unknown")}
- Name: {item.get("name", "Unknown")}
- Price: ${item.get("price", 0.00)}
- Category: {item.get("category", "Unknown")}"""

    agent = await client.agents.create(
        memory_blocks=[
            CreateBlock(
                label="persona",
                value="You are a helpful inventory management assistant.",
            ),
            CreateBlock(
                label="item_context",
                value=item_context,
            ),
        ],
        model="openai/gpt-4.1-mini",
        embedding="openai/text-embedding-3-small",
        tool_ids=[tool.id],
        include_base_tools=False,
    )

    return agent.id
