import json
from typing import Any, Dict, List, Optional

from letta_client.types import AgentState

from letta_evals.decorators import extractor
from letta_evals.models import LettaMessageUnion


def _memory_blocks_payload(agent_state: Optional[AgentState]) -> Dict[str, Any]:
    """Return ALL memory blocks (post-run) for judge evaluation."""
    if not agent_state or not getattr(agent_state, "memory", None):
        return {"error": "No agent_state.memory available."}

    blocks: List[Any] = getattr(agent_state.memory, "blocks", None) or []

    out_blocks: List[Dict[str, Any]] = []
    for b in blocks:
        out_blocks.append(
            {
                "label": getattr(b, "label", None) or getattr(b, "name", None),
                "id": getattr(b, "id", None),
                "read_only": bool(getattr(b, "read_only", False)),
                "limit": getattr(b, "limit", None),
                "description": getattr(b, "description", None),
                "value": getattr(b, "value", None),
            }
        )

    return {"memory_blocks": out_blocks}


def _format_trajectory(trajectory: List[List[LettaMessageUnion]]) -> List[Dict[str, Any]]:
    """Format trajectory for judge evaluation - includes tool calls and messages."""
    formatted = []
    for turn in trajectory:
        for msg in turn:
            entry = {"type": type(msg).__name__}
            
            # Extract relevant fields based on message type
            if hasattr(msg, "content") and msg.content:
                entry["content"] = msg.content
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "name": tc.function.name if hasattr(tc, "function") else getattr(tc, "name", None),
                        "arguments": tc.function.arguments if hasattr(tc, "function") else getattr(tc, "arguments", None),
                    }
                    for tc in msg.tool_calls
                ]
            if hasattr(msg, "tool_call") and msg.tool_call:
                tc = msg.tool_call
                entry["tool_call"] = {
                    "name": tc.function.name if hasattr(tc, "function") else getattr(tc, "name", None),
                    "arguments": tc.function.arguments if hasattr(tc, "function") else getattr(tc, "arguments", None),
                }
            if hasattr(msg, "tool_return") and msg.tool_return:
                # Truncate long tool returns
                tool_return = msg.tool_return
                if len(tool_return) > 500:
                    tool_return = tool_return[:500] + "... [truncated]"
                entry["tool_return"] = tool_return
            
            formatted.append(entry)
    return formatted


@extractor
def defrag_evidence_extractor(
    trajectory: List[List[LettaMessageUnion]], config: dict, agent_state: Optional[AgentState] = None
) -> str:
    """Extractor used for memory-defrag judging.

    Returns both trajectory (for evaluating skill loading, memory edits, completion summary)
    and memory blocks (for evaluating final memory quality).
    """
    payload = {
        "trajectory": _format_trajectory(trajectory),
        **_memory_blocks_payload(agent_state),
    }
    return json.dumps(payload, indent=2)
