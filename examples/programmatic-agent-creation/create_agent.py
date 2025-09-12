from typing import List, Type

from letta_client import AsyncLetta, CreateBlock
from letta_client.client import BaseTool
from pydantic import BaseModel

from letta_evals.targets.agent_factory import AgentFactory


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


class InventoryAgentFactory(AgentFactory):
    """Factory that creates an inventory management agent with custom tools."""

    async def create(self, client: AsyncLetta) -> str:
        """Create an inventory management agent using the Letta SDK."""
        # Use the tool if it already exists
        tools = await client.tools.list(name="manage_inventory")
        if tools:
            tool = tools[0]
        else:
            tool = await client.tools.add(
                tool=ManageInventoryTool(),
            )

        agent = await client.agents.create(
            memory_blocks=[
                CreateBlock(
                    label="persona",
                    value="You are a helpful inventory management assistant.",
                ),
            ],
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
            tool_ids=[tool.id],
            include_base_tools=False,
        )

        return agent.id
