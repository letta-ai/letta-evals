from typing import List, Type

# SDK v1.0+: BaseTool moved out of letta_client.client
from letta_client.types.tool import BaseTool
from pydantic import BaseModel

TEST_TOOL_NAME = "evals_ci_manage_inventory_tool"


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
    name: str = TEST_TOOL_NAME
    args_schema: Type[BaseModel] = InventoryEntryData
    description: str = "Update inventory catalogue with a new data entry"
    tags: List[str] = ["inventory", "shop"]

    def run(self, data: InventoryEntry, quantity_change: int) -> str:
        return f"Updated inventory for {data.item.name} with a quantity change of {quantity_change}"
