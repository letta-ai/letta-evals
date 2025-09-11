import json
import re
from typing import List

from letta_client import AssistantMessage, LettaMessageUnion

from letta_evals.graders.extractors.base import SubmissionExtractor
from letta_evals.graders.extractors.utils import (
    flatten_content,
    get_assistant_messages,
    get_last_turn_messages,
    get_tool_calls,
)


class LastAssistantExtractor(SubmissionExtractor):
    """Extract the last assistant message content."""

    def extract(self, trajectory: List[List[LettaMessageUnion]]) -> str:
        messages = get_assistant_messages(trajectory)
        if not messages:
            return ""
        return flatten_content(messages[-1].content)


class FirstAssistantExtractor(SubmissionExtractor):
    """Extract the first assistant message content."""

    def extract(self, trajectory: List[List[LettaMessageUnion]]) -> str:
        messages = get_assistant_messages(trajectory)
        if not messages:
            return ""
        return flatten_content(messages[0].content)


class AllAssistantExtractor(SubmissionExtractor):
    """Concatenate all assistant messages."""

    def extract(self, trajectory: List[List[LettaMessageUnion]]) -> str:
        messages = get_assistant_messages(trajectory)
        contents = [flatten_content(msg.content) for msg in messages]
        separator = self.config.get("separator", "\n")
        return separator.join(contents)


class LastTurnExtractor(SubmissionExtractor):
    """Extract all assistant messages from the last turn."""

    def extract(self, trajectory: List[List[LettaMessageUnion]]) -> str:
        messages = get_last_turn_messages(trajectory, AssistantMessage)
        contents = [flatten_content(msg.content) for msg in messages]
        separator = self.config.get("separator", "\n")
        return separator.join(contents)


class PatternExtractor(SubmissionExtractor):
    """Extract content matching a regex pattern."""

    def extract(self, trajectory: List[List[LettaMessageUnion]]) -> str:
        pattern = self.config["pattern"]
        group = self.config.get("group", 0)
        search_all = self.config.get("search_all", False)

        messages = get_assistant_messages(trajectory)

        for msg in reversed(messages):
            content = flatten_content(msg.content)
            if search_all:
                matches = re.findall(pattern, content)
                if matches:
                    if isinstance(matches[0], tuple):
                        return " ".join(m[group] for m in matches)
                    else:
                        return " ".join(matches)
            else:
                match = re.search(pattern, content)
                if match:
                    return match.group(group)

        return ""


class JSONExtractor(SubmissionExtractor):
    """Extract JSON fields from structured responses."""

    def extract(self, trajectory: List[List[LettaMessageUnion]]) -> str:
        field = self.config["field"]

        messages = get_assistant_messages(trajectory)

        for msg in reversed(messages):
            content = flatten_content(msg.content)
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
                fields = field.split(".")
                value = data
                for f in fields:
                    value = value[f]

                return str(value)

        return ""


class ToolOutputExtractor(SubmissionExtractor):
    """Extract output from specific tool calls."""

    def extract(self, trajectory: List[List[LettaMessageUnion]]) -> str:
        tool_name = self.config["tool_name"]
        field = self.config.get("field")

        tool_pairs = get_tool_calls(trajectory)

        for call, ret in reversed(tool_pairs):
            for tool_call in call.tool_calls:
                if tool_call.function.name == tool_name:
                    output = ret.tool_return
                    if field:
                        return str(getattr(output, field))
                    else:
                        return str(output)

        return ""


class AfterMarkerExtractor(SubmissionExtractor):
    """Extract content after a specific marker."""

    def extract(self, trajectory: List[List[LettaMessageUnion]]) -> str:
        marker = self.config["marker"]
        include_marker = self.config.get("include_marker", False)

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
