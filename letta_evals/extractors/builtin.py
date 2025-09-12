import re
from typing import List

from letta_client import AssistantMessage, LettaMessageUnion, ToolCallMessage, ToolReturnMessage

from letta_evals.decorators import extractor
from letta_evals.extractors.utils import (
    flatten_content,
    get_assistant_messages,
    get_last_turn_messages,
)


@extractor
def last_assistant(trajectory: List[List[LettaMessageUnion]], config: dict) -> str:
    """Extract the last assistant message content."""
    messages = get_assistant_messages(trajectory)
    if not messages:
        return ""
    return flatten_content(messages[-1].content)


@extractor
def first_assistant(trajectory: List[List[LettaMessageUnion]], config: dict) -> str:
    """Extract the first assistant message content."""
    messages = get_assistant_messages(trajectory)
    if not messages:
        return ""
    return flatten_content(messages[0].content)


@extractor
def all_assistant(trajectory: List[List[LettaMessageUnion]], config: dict) -> str:
    """Concatenate all assistant messages."""
    messages = get_assistant_messages(trajectory)
    contents = [flatten_content(msg.content) for msg in messages]
    separator = config.get("separator", "\n")
    return separator.join(contents)


@extractor
def last_turn(trajectory: List[List[LettaMessageUnion]], config: dict) -> str:
    """Extract all assistant messages from the last turn."""
    messages = get_last_turn_messages(trajectory, AssistantMessage)
    contents = [flatten_content(msg.content) for msg in messages]
    separator = config.get("separator", "\n")
    return separator.join(contents)


@extractor
def pattern(trajectory: List[List[LettaMessageUnion]], config: dict) -> str:
    """Extract content matching a regex pattern."""
    pattern_str = config["pattern"]
    group = config.get("group", 0)
    search_all = config.get("search_all", False)

    messages = get_assistant_messages(trajectory)

    for msg in reversed(messages):
        content = flatten_content(msg.content)
        if search_all:
            matches = re.findall(pattern_str, content)
            if matches:
                if isinstance(matches[0], tuple):
                    return " ".join(m[group] for m in matches)
                else:
                    return " ".join(matches)
        else:
            match = re.search(pattern_str, content)
            if match:
                return match.group(group)

    return ""


@extractor
def tool_arguments(trajectory: List[List[LettaMessageUnion]], config: dict) -> str:
    """Extract arguments from specific tool calls."""
    tool_name = config["tool_name"]

    for turn in trajectory:
        for message in turn:
            if isinstance(message, ToolCallMessage):
                if message.tool_call.name == tool_name:
                    return message.tool_call.arguments

    return "{}"


@extractor
def tool_output(trajectory: List[List[LettaMessageUnion]], config: dict) -> str:
    """Extract output from specific tool calls."""
    tool_name = config["tool_name"]

    # find the tool call first
    tool_call_id = None
    for turn in trajectory:
        for message in turn:
            if isinstance(message, ToolCallMessage):
                if message.tool_call.name == tool_name:
                    tool_call_id = message.tool_call.tool_call_id
                    break
            if tool_call_id:
                break
        if tool_call_id:
            break

    # if we found a matching tool call, find its return
    if tool_call_id:
        for turn in trajectory:
            for message in turn:
                if isinstance(message, ToolReturnMessage) and message.tool_call_id == tool_call_id:
                    return message.tool_return

    return ""


@extractor
def after_marker(trajectory: List[List[LettaMessageUnion]], config: dict) -> str:
    """Extract content after a specific marker."""
    marker = config["marker"]
    include_marker = config.get("include_marker", False)

    messages = get_assistant_messages(trajectory)

    for msg in reversed(messages):
        content = flatten_content(msg.content)
        idx = content.find(marker)
        if idx >= 0:
            if include_marker:
                return content[idx:].strip()
            else:
                return content[idx + len(marker) :].strip()

    return ""
