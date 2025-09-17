from typing import List, Type

from letta_client import AsyncLetta
from letta_client.client import BaseTool
from pydantic import BaseModel

from letta_evals.decorators import suite_setup


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


@suite_setup
async def prepare_evaluation(client: AsyncLetta) -> None:
    """Set up the evaluation environment by creating required tools."""
    tools = await client.tools.list(name="manage_inventory")
    if not tools:
        await client.tools.add(tool=ManageInventoryTool())
        print("Created manage_inventory tool")
    else:
        print("manage_inventory tool already exists")
