"""Unit tests for built-in extractors."""

from datetime import datetime, timezone

from letta_client.types.agents import AssistantMessage, ToolCallMessage

from letta_evals.extractors.builtin import (
    all_assistant,
    first_assistant,
    last_assistant,
    last_turn,
    pattern,
    tool_arguments,
)

_FAKE_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _msg(content: str, id: str = "msg-0") -> AssistantMessage:
    return AssistantMessage(id=id, message_type="assistant_message", date=_FAKE_DATE, content=content)


def _traj(*turns: list) -> list[list]:
    """Build a trajectory from turn lists."""
    return list(turns)


# ── last_assistant ──


class TestLastAssistant:
    def test_single_message(self):
        assert last_assistant(_traj([_msg("hello")]), {}) == "hello"

    def test_multiple_messages_returns_last(self):
        traj = _traj([_msg("first", "m1"), _msg("second", "m2")])
        assert last_assistant(traj, {}) == "second"

    def test_multiple_turns_returns_last(self):
        traj = _traj([_msg("turn1", "m1")], [_msg("turn2", "m2")])
        assert last_assistant(traj, {}) == "turn2"

    def test_empty_trajectory(self):
        assert last_assistant([], {}) == ""

    def test_no_assistant_messages(self):
        traj = _traj([[]])
        assert last_assistant(traj, {}) == ""


# ── first_assistant ──


class TestFirstAssistant:
    def test_single_message(self):
        assert first_assistant(_traj([_msg("hello")]), {}) == "hello"

    def test_multiple_messages_returns_first(self):
        traj = _traj([_msg("first", "m1"), _msg("second", "m2")])
        assert first_assistant(traj, {}) == "first"

    def test_empty_trajectory(self):
        assert first_assistant([], {}) == ""


# ── all_assistant ──


class TestAllAssistant:
    def test_concatenates_messages(self):
        traj = _traj([_msg("hello", "m1"), _msg("world", "m2")])
        assert all_assistant(traj, {}) == "hello\nworld"

    def test_custom_separator(self):
        traj = _traj([_msg("a", "m1"), _msg("b", "m2")])
        assert all_assistant(traj, {"separator": " | "}) == "a | b"

    def test_across_turns(self):
        traj = _traj([_msg("turn1", "m1")], [_msg("turn2", "m2")])
        assert all_assistant(traj, {}) == "turn1\nturn2"

    def test_empty_trajectory(self):
        assert all_assistant([], {}) == ""


# ── last_turn ──


class TestLastTurn:
    def test_single_turn(self):
        traj = _traj([_msg("hello")])
        assert last_turn(traj, {}) == "hello"

    def test_multiple_turns_returns_last(self):
        traj = _traj([_msg("turn1", "m1")], [_msg("turn2a", "m2"), _msg("turn2b", "m3")])
        assert last_turn(traj, {}) == "turn2a\nturn2b"

    def test_empty_trajectory(self):
        assert last_turn([], {}) == ""


# ── pattern ──


class TestPattern:
    def test_simple_match(self):
        traj = _traj([_msg("The answer is 42")])
        assert pattern(traj, {"pattern": r"\d+"}) == "42"

    def test_group_capture(self):
        traj = _traj([_msg("name: Alice, age: 30")])
        assert pattern(traj, {"pattern": r"name: (\w+)", "group": 1}) == "Alice"

    def test_no_match(self):
        traj = _traj([_msg("no numbers here")])
        assert pattern(traj, {"pattern": r"\d+"}) == ""

    def test_search_all(self):
        traj = _traj([_msg("1 2 3")])
        assert pattern(traj, {"pattern": r"\d+", "search_all": True}) == "1 2 3"

    def test_searches_from_last_message(self):
        traj = _traj([_msg("old: 10", "m1"), _msg("new: 20", "m2")])
        assert pattern(traj, {"pattern": r"\d+"}) == "20"

    def test_empty_trajectory(self):
        assert pattern([], {"pattern": r"\d+"}) == ""


# ── tool_arguments ──


class TestToolArguments:
    def _tool_call_msg(self, tool_name: str, arguments: str) -> ToolCallMessage:
        from letta_client.types.agents import ToolCall

        return ToolCallMessage(
            id="tc-0",
            message_type="tool_call_message",
            date=_FAKE_DATE,
            tool_call=ToolCall(name=tool_name, arguments=arguments, tool_call_id="call-0"),
        )

    def test_extracts_matching_tool(self):
        traj = _traj([self._tool_call_msg("search", '{"query": "hello"}')])
        assert tool_arguments(traj, {"tool_name": "search"}) == '{"query": "hello"}'

    def test_no_matching_tool(self):
        traj = _traj([self._tool_call_msg("search", '{"q": "hi"}')])
        assert tool_arguments(traj, {"tool_name": "other_tool"}) == "{}"

    def test_empty_trajectory(self):
        assert tool_arguments([], {"tool_name": "search"}) == "{}"
