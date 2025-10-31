# Programmatic Agent Creation Example

This example demonstrates dynamic agent creation using Python scripts instead of pre-saved agent files.

## What This Example Shows

- Using the `@agent_factory` decorator to create agents programmatically for each sample
- Using the `@suite_setup` decorator for pre-evaluation environment preparation
- Customizing agents per sample using `agent_args` from the dataset
- Programmatic tool registration and agent configuration

## Key Takeaway

Programmatic agent creation is useful when you need to test agent creation logic itself, customize agents per sample, or don't have pre-saved agent files. In this example, each sample gets a unique agent with item-specific memory blocks.

## Running This Example

### Local Setup (Default)

Start your local Letta server:
```bash
letta server
```

Then run the evaluation:
```bash
cd examples/programmatic-agent-creation
letta-evals run suite.yaml
```

### Letta Cloud Setup

Set these environment variables:
```bash
export LETTA_API_KEY=your-api-key
export LETTA_PROJECT_ID=your-project-id
```

Update `base_url` in `suite.yaml`:
```yaml
target:
  base_url: https://api.letta.com/
```

Then run the evaluation as above.

## Code Walkthrough

### Step 1: Suite Setup (`setup.py:prepare_evaluation`)

The `@suite_setup` decorator runs once before the evaluation begins:

```python
@suite_setup
async def prepare_evaluation(client: AsyncLetta) -> None:
    """Set up the evaluation environment by creating required tools."""
    tools = await client.tools.list(name=TEST_TOOL_NAME)
    if not tools:
        await client.tools.add(tool=ManageInventoryTool())
        print(f"Created {TEST_TOOL_NAME} tool")
    else:
        print(f"{TEST_TOOL_NAME} tool already exists")
```

**Key points:**
- The `@suite_setup` decorator ensures this runs once before any agents are created
- Receives an `AsyncLetta` client instance
- Can be used for any pre-evaluation setup (tools, data loading, etc.)
- Idempotent check prevents duplicate tool creation
- Supports three signatures:
  - `() -> None` - No parameters
  - `(client: AsyncLetta) -> None` - With client (shown above)
  - `(client: AsyncLetta, model_name: str) -> None` - With client and model name (runs once per model when testing multiple models)

### Step 2: Agent Factory (`create_agent.py:create_inventory_agent`)

The `@agent_factory` decorator creates a fresh agent for each sample:

```python
@agent_factory
async def create_inventory_agent(client: AsyncLetta, sample: Sample) -> str:
    """Create an inventory management agent using the Letta SDK.

    The agent is customized with item details from sample.agent_args.
    """
    tools = await client.tools.list(name=TEST_TOOL_NAME)
    if not tools:
        raise RuntimeError(f"Tool '{TEST_TOOL_NAME}' not found. Please ensure setup has been run.")
    tool = tools[0]

    item = sample.agent_args["item"]
    item_context = f"""Target Item Details:
- SKU: {item.get("sku", "Unknown")}
- Name: {item.get("name", "Unknown")}
- Price: ${item.get("price", 0.00)}
- Category: {item.get("category", "Unknown")}"""

    agent = await client.agents.create(
        name="inventory-assistant",
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
        agent_type="letta_v1_agent",
        model="openai/gpt-4.1-mini",
        embedding="openai/text-embedding-3-small",
        tool_ids=[tool.id],
        include_base_tools=False,
        project_id=os.environ.get("LETTA_PROJECT_ID"),
    )

    return agent.id
```

**Key points:**
- The `@agent_factory` decorator is called once per sample
- Receives both the `client` and the current `sample`
- Extracts `agent_args` from the sample to customize the agent
- Creates unique memory blocks with item-specific context
- Returns the agent ID (string) for the evaluation runner to use

### Step 3: Dataset with Agent Args

Each sample in `dataset.jsonl` includes `agent_args` for customization:

```json
{
  "input": "Add 10 units to inventory",
  "ground_truth": "Updated inventory for Widget A with a quantity change of \\d+",
  "agent_args": {
    "item": {
      "sku": "SKU-123",
      "name": "Widget A",
      "price": 19.99,
      "category": "electronics"
    }
  }
}
```

**Key points:**
- `agent_args` is passed to the agent factory via `sample.agent_args`
- Allows per-sample customization (different products, configurations, etc.)
- The factory uses these args to create item-specific memory blocks

### Suite Configuration

In `suite.yaml`, specify the setup and factory scripts:

```yaml
setup_script: setup.py:prepare_evaluation
target:
  kind: letta_agent
  agent_script: create_agent.py:create_inventory_agent
```
