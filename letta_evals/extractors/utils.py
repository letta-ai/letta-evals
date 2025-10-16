from typing import Any, List

from letta_client import (
    AssistantMessage,
    LettaMessageUnion,
    ToolCallMessage,
    ToolReturnMessage,
)


def flatten_content(content: Any) -> str:
    """Flatten message content to string."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = [item.text for item in content]
        return " ".join(parts)
    else:
        raise ValueError(f"Unexpected content type: {type(content)}")


def get_assistant_messages(trajectory: List[List[LettaMessageUnion]]) -> List[AssistantMessage]:
    """Extract all assistant messages from trajectory."""
    messages = []
    for turn in trajectory:
        for msg in turn:
            if isinstance(msg, AssistantMessage):
                messages.append(msg)
    return messages


def get_tool_calls(trajectory: List[List[LettaMessageUnion]]) -> List[tuple[ToolCallMessage, ToolReturnMessage]]:
    """Extract all tool call/return pairs from trajectory."""
    pairs = []
    for turn in trajectory:
        i = 0
        while i < len(turn):
            if isinstance(turn[i], ToolCallMessage):
                for j in range(i + 1, len(turn)):
                    if isinstance(turn[j], ToolReturnMessage):
                        pairs.append((turn[i], turn[j]))
                        break
            i += 1
    return pairs


def get_messages_by_type(trajectory: List[List[LettaMessageUnion]], message_type: type) -> List[LettaMessageUnion]:
    """Filter messages by type."""
    messages = []
    for turn in trajectory:
        for msg in turn:
            if isinstance(msg, message_type):
                messages.append(msg)
    return messages


def get_last_turn_messages(
    trajectory: List[List[LettaMessageUnion]], message_type: type = AssistantMessage
) -> List[LettaMessageUnion]:
    """Get all messages of a given type from the last turn."""
    if not trajectory:
        return []

    last_turn = trajectory[-1]
    return [msg for msg in last_turn if isinstance(msg, message_type)]
