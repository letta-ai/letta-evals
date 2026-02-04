"""Tests for AgentFile wrapper."""

import json
import tempfile
from pathlib import Path

import pytest
from agentfile import AgentFile


@pytest.fixture
def sample_af_data():
    """Minimal valid .af structure with a conversation."""
    return {
        "agents": [
            {
                "name": "test-agent",
                "memory_blocks": [],
                "tools": [],
                "tool_ids": [],
                "source_ids": [],
                "block_ids": [],
                "tool_rules": [],
                "tags": [],
                "system": "You are a test assistant.",
                "agent_type": "letta_v1_agent",
                "llm_config": {
                    "model": "gpt-4o-mini",
                    "model_endpoint_type": "openai",
                    "handle": "openai/gpt-4o-mini",
                    "context_window": 128000,
                },
                "embedding_config": {
                    "embedding_model": "text-embedding-3-small",
                    "handle": "openai/text-embedding-3-small",
                },
                "id": "agent-0",
                "in_context_message_ids": [
                    "message-0",
                    "message-1",
                    "message-2",
                    "message-3",
                    "message-4",
                    "message-5",
                ],
                "messages": [
                    {
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a test assistant."}],
                        "id": "message-0",
                        "agent_id": "agent-0",
                        "tool_calls": None,
                        "tool_returns": [],
                        "created_at": "2026-01-01T00:00:00+00:00",
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "text", "text": "Hello"}],
                        "id": "message-1",
                        "agent_id": "agent-0",
                        "tool_calls": None,
                        "tool_returns": [],
                        "created_at": "2026-01-01T00:00:01+00:00",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hi there!"}],
                        "id": "message-2",
                        "agent_id": "agent-0",
                        "tool_calls": None,
                        "tool_returns": [],
                        "created_at": "2026-01-01T00:00:02+00:00",
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "text", "text": "What is 2+2?"}],
                        "id": "message-3",
                        "agent_id": "agent-0",
                        "tool_calls": None,
                        "tool_returns": [],
                        "created_at": "2026-01-01T00:00:03+00:00",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "id": "message-4",
                        "agent_id": "agent-0",
                        "tool_calls": [{"id": "call-1", "function": {"name": "calculator", "arguments": "{}"}}],
                        "tool_returns": [],
                        "created_at": "2026-01-01T00:00:04+00:00",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "2+2 equals 4."}],
                        "id": "message-5",
                        "agent_id": "agent-0",
                        "tool_calls": None,
                        "tool_returns": [],
                        "created_at": "2026-01-01T00:00:05+00:00",
                    },
                ],
            }
        ],
        "groups": [],
        "blocks": [],
        "files": [],
        "sources": [],
        "tools": [],
        "mcp_servers": [],
        "metadata": {"revision_id": "test"},
        "created_at": "2026-01-01T00:00:00+00:00",
    }


@pytest.fixture
def sample_af(sample_af_data):
    """AgentFile instance from sample data."""
    return AgentFile(sample_af_data)


@pytest.fixture
def temp_af_file(sample_af_data):
    """Temporary .af file on disk."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".af", delete=False) as f:
        json.dump(sample_af_data, f)
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


class TestLoadSave:
    def test_load(self, temp_af_file):
        af = AgentFile.load(temp_af_file)
        assert af.name == "test-agent"
        assert af.num_messages == 6

    def test_save(self, sample_af):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".af", delete=False) as f:
            path = Path(f.name)

        try:
            sample_af.save(path)
            reloaded = AgentFile.load(path)
            assert reloaded.name == sample_af.name
            assert reloaded.num_messages == sample_af.num_messages
        finally:
            path.unlink(missing_ok=True)

    def test_roundtrip_preserves_data(self, sample_af):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".af", delete=False) as f:
            path = Path(f.name)

        try:
            sample_af.save(path)
            reloaded = AgentFile.load(path)
            assert reloaded.data == sample_af.data
        finally:
            path.unlink(missing_ok=True)


class TestCreateEmpty:
    def test_create_empty_defaults(self):
        af = AgentFile.create_empty()
        assert af.name == "empty-agent"
        assert af.num_messages == 1  # system message only
        assert af.messages[0]["role"] == "system"

    def test_create_empty_custom_name(self):
        af = AgentFile.create_empty(name="my-agent")
        assert af.name == "my-agent"

    def test_create_empty_custom_model(self):
        af = AgentFile.create_empty(model_handle="anthropic/claude-3-5-sonnet")
        assert "claude-3-5-sonnet" in af.model

    def test_create_empty_custom_system_prompt(self):
        af = AgentFile.create_empty(system_prompt="Be concise.")
        assert af.agent["system"] == "Be concise."
        assert "Be concise." in af._extract_content(af.messages[0])

    def test_create_empty_valid_structure(self):
        af = AgentFile.create_empty()
        # Should have all required top-level keys
        assert "agents" in af.data
        assert "blocks" in af.data
        assert "tools" in af.data
        assert "metadata" in af.data


class TestRevert:
    def test_revert_one_step(self, sample_af):
        """Revert(1) should remove last user message and its responses."""
        # Original: system, user1, assistant1, user2, assistant2(tool), assistant2(response)
        # After revert(1): system, user1, assistant1
        reverted = sample_af.revert(1)
        assert reverted.num_messages == 3
        assert reverted.messages[-1]["role"] == "assistant"
        assert "Hi there" in reverted._extract_content(reverted.messages[-1])

    def test_revert_two_steps(self, sample_af):
        """Revert(2) should remove last two user turns."""
        # After revert(2): system only
        reverted = sample_af.revert(2)
        assert reverted.num_messages == 1
        assert reverted.messages[0]["role"] == "system"

    def test_revert_more_than_available(self, sample_af):
        """Revert(N) where N > turns should keep only system message."""
        reverted = sample_af.revert(100)
        assert reverted.num_messages == 1
        assert reverted.messages[0]["role"] == "system"

    def test_revert_immutable(self, sample_af):
        """Original should be unchanged after revert."""
        original_count = sample_af.num_messages
        _ = sample_af.revert(1)
        assert sample_af.num_messages == original_count

    def test_revert_updates_in_context_ids(self, sample_af):
        """in_context_message_ids should match remaining messages."""
        reverted = sample_af.revert(1)
        msg_ids = [m["id"] for m in reverted.messages]
        assert reverted.agent["in_context_message_ids"] == msg_ids


class TestRevertMessages:
    def test_revert_messages_one(self, sample_af):
        """Remove exactly 1 message."""
        reverted = sample_af.revert_messages(1)
        assert reverted.num_messages == 5

    def test_revert_messages_multiple(self, sample_af):
        """Remove multiple messages."""
        reverted = sample_af.revert_messages(3)
        assert reverted.num_messages == 3

    def test_revert_messages_preserves_system(self, sample_af):
        """Should always keep at least system message."""
        reverted = sample_af.revert_messages(100)
        assert reverted.num_messages == 1
        assert reverted.messages[0]["role"] == "system"

    def test_revert_messages_immutable(self, sample_af):
        """Original unchanged."""
        original_count = sample_af.num_messages
        _ = sample_af.revert_messages(2)
        assert sample_af.num_messages == original_count


class TestDiff:
    def test_diff_same(self, sample_af):
        """Diff of same file shows no changes."""
        diff = sample_af.diff(sample_af)
        assert "6 -> 6" in diff
        assert "New messages" not in diff

    def test_diff_added_messages(self, sample_af):
        """Diff shows new messages."""
        # Create modified version with extra message
        import copy

        data2 = copy.deepcopy(sample_af.data)
        new_msg = {
            "type": "message",
            "role": "user",
            "content": [{"type": "text", "text": "New question"}],
            "id": "message-6",
            "agent_id": "agent-0",
            "tool_calls": None,
            "tool_returns": [],
            "created_at": "2026-01-01T00:00:06+00:00",
        }
        data2["agents"][0]["messages"].append(new_msg)
        data2["agents"][0]["in_context_message_ids"].append("message-6")

        af2 = AgentFile(data2)
        diff = sample_af.diff(af2)

        assert "6 -> 7" in diff
        assert "New messages (1)" in diff
        assert "New question" in diff

    def test_diff_reverted(self, sample_af):
        """Diff after revert shows fewer messages."""
        reverted = sample_af.revert(1)
        diff = reverted.diff(sample_af)
        assert "3 -> 6" in diff


class TestProperties:
    def test_name(self, sample_af):
        assert sample_af.name == "test-agent"

    def test_model(self, sample_af):
        assert sample_af.model == "openai/gpt-4o-mini"

    def test_num_messages(self, sample_af):
        assert sample_af.num_messages == 6

    def test_messages(self, sample_af):
        assert len(sample_af.messages) == 6
        assert sample_af.messages[0]["role"] == "system"

    def test_agent(self, sample_af):
        assert sample_af.agent["name"] == "test-agent"


class TestGetConversation:
    def test_excludes_system_by_default(self, sample_af):
        conv = sample_af.get_conversation()
        assert len(conv) == 5
        assert conv[0]["role"] == "user"

    def test_includes_system_when_requested(self, sample_af):
        conv = sample_af.get_conversation(include_system=True)
        assert len(conv) == 6
        assert conv[0]["role"] == "system"

    def test_extracts_content(self, sample_af):
        conv = sample_af.get_conversation()
        assert conv[0]["content"] == "Hello"
        assert conv[1]["content"] == "Hi there!"

    def test_includes_tool_calls(self, sample_af):
        conv = sample_af.get_conversation()
        # message-4 has tool_calls
        tool_msg = [m for m in conv if m.get("tool_calls")]
        assert len(tool_msg) == 1


class TestRepr:
    def test_repr(self, sample_af):
        r = repr(sample_af)
        assert "test-agent" in r
        assert "6" in r  # message count
        assert "gpt-4o-mini" in r


class TestMinTurns:
    def test_min_turns_true(self, sample_af):
        assert sample_af.min_turns(2) is True

    def test_min_turns_false(self, sample_af):
        assert sample_af.min_turns(10) is False

    def test_min_turns_exact(self, sample_af):
        # sample_af has 2 user turns
        assert sample_af.min_turns(2) is True
        assert sample_af.min_turns(3) is False


class TestTruncateToolOutputs:
    @pytest.fixture
    def af_with_tool_output(self, sample_af_data):
        """AgentFile with a long tool output."""
        import copy
        data = copy.deepcopy(sample_af_data)
        # Add a tool message with long content
        tool_msg = {
            "type": "message",
            "role": "tool",
            "content": [{"type": "text", "text": "x" * 1000}],
            "id": "message-6",
            "name": "some_tool",
            "agent_id": "agent-0",
            "tool_calls": None,
            "tool_returns": [],
            "created_at": "2026-01-01T00:00:06+00:00",
        }
        data["agents"][0]["messages"].append(tool_msg)
        data["agents"][0]["in_context_message_ids"].append("message-6")
        return AgentFile(data)

    def test_truncates_long_output(self, af_with_tool_output):
        truncated = af_with_tool_output.truncate_tool_outputs(100)
        tool_msg = [m for m in truncated.messages if m["role"] == "tool"][0]
        content = truncated._extract_content(tool_msg)
        assert len(content) < 200  # 100 + "... [truncated]"
        assert "[truncated]" in content

    def test_preserves_short_output(self, sample_af):
        truncated = sample_af.truncate_tool_outputs(100)
        assert truncated.num_messages == sample_af.num_messages

    def test_immutable(self, af_with_tool_output):
        original_len = len(af_with_tool_output._extract_content(af_with_tool_output.messages[-1]))
        _ = af_with_tool_output.truncate_tool_outputs(100)
        new_len = len(af_with_tool_output._extract_content(af_with_tool_output.messages[-1]))
        assert original_len == new_len


class TestRandomizeIdentity:
    def test_changes_agent_id(self, sample_af):
        randomized = sample_af.randomize_identity()
        assert randomized.agent["id"] != sample_af.agent["id"]

    def test_changes_message_ids(self, sample_af):
        randomized = sample_af.randomize_identity()
        orig_ids = {m["id"] for m in sample_af.messages}
        new_ids = {m["id"] for m in randomized.messages}
        assert orig_ids != new_ids

    def test_changes_agent_name(self, sample_af):
        randomized = sample_af.randomize_identity()
        assert randomized.name != sample_af.name

    def test_seed_reproducible(self, sample_af):
        r1 = sample_af.randomize_identity(seed=42)
        r2 = sample_af.randomize_identity(seed=42)
        assert r1.agent["id"] == r2.agent["id"]
        assert r1.name == r2.name

    def test_immutable(self, sample_af):
        original_id = sample_af.agent["id"]
        _ = sample_af.randomize_identity()
        assert sample_af.agent["id"] == original_id


class TestStripSystemReminders:
    @pytest.fixture
    def af_with_reminders(self, sample_af_data):
        """AgentFile with system reminder messages."""
        import copy
        data = copy.deepcopy(sample_af_data)
        # Add system reminder message
        reminder_msg = {
            "type": "message",
            "role": "user",
            "content": [{"type": "text", "text": "<system-reminder>MEMORY CHECK: Review this...</system-reminder>"}],
            "id": "message-6",
            "agent_id": "agent-0",
            "tool_calls": None,
            "tool_returns": [],
            "created_at": "2026-01-01T00:00:06+00:00",
        }
        data["agents"][0]["messages"].append(reminder_msg)
        data["agents"][0]["in_context_message_ids"].append("message-6")
        return AgentFile(data)

    def test_removes_system_reminders(self, af_with_reminders):
        stripped = af_with_reminders.strip_system_reminders()
        assert stripped.num_messages == af_with_reminders.num_messages - 1

    def test_removes_memory_check(self, sample_af_data):
        import copy
        data = copy.deepcopy(sample_af_data)
        data["agents"][0]["messages"].append({
            "type": "message",
            "role": "user",
            "content": [{"type": "text", "text": "MEMORY CHECK: Do something"}],
            "id": "message-6",
            "agent_id": "agent-0",
            "tool_calls": None,
            "tool_returns": [],
            "created_at": "2026-01-01T00:00:06+00:00",
        })
        af = AgentFile(data)
        stripped = af.strip_system_reminders()
        assert stripped.num_messages == 6

    def test_preserves_normal_messages(self, sample_af):
        stripped = sample_af.strip_system_reminders()
        assert stripped.num_messages == sample_af.num_messages

    def test_immutable(self, af_with_reminders):
        original_count = af_with_reminders.num_messages
        _ = af_with_reminders.strip_system_reminders()
        assert af_with_reminders.num_messages == original_count


class TestToTrainingFormat:
    def test_returns_list(self, sample_af):
        pairs = sample_af.to_training_format()
        assert isinstance(pairs, list)

    def test_correct_number_of_pairs(self, sample_af):
        # sample_af has 2 user turns, both with responses
        pairs = sample_af.to_training_format()
        assert len(pairs) == 2

    def test_pair_structure(self, sample_af):
        pairs = sample_af.to_training_format()
        pair = pairs[0]
        assert "id" in pair
        assert "context" in pair
        assert "target" in pair
        assert "metadata" in pair

    def test_context_ends_with_user(self, sample_af):
        pairs = sample_af.to_training_format()
        for pair in pairs:
            last_ctx = pair["context"][-1]
            assert last_ctx["role"] == "user"

    def test_target_has_assistant(self, sample_af):
        pairs = sample_af.to_training_format()
        for pair in pairs:
            roles = [m["role"] for m in pair["target"]]
            assert "assistant" in roles

    def test_metadata_tracks_tools(self, sample_af):
        pairs = sample_af.to_training_format()
        # Second turn uses calculator tool
        pair = pairs[1]
        assert "calculator" in pair["metadata"]["tools_used"]

    def test_source_in_id(self, sample_af):
        pairs = sample_af.to_training_format(source="test-source")
        assert pairs[0]["id"].startswith("test-source")

    def test_truncates_tool_output(self, sample_af_data):
        import copy
        data = copy.deepcopy(sample_af_data)
        # Add tool response with long content
        data["agents"][0]["messages"].insert(5, {
            "type": "message",
            "role": "tool",
            "content": [{"type": "text", "text": "x" * 2000}],
            "id": "message-tool",
            "name": "calculator",
            "agent_id": "agent-0",
            "tool_calls": None,
            "tool_returns": [],
            "created_at": "2026-01-01T00:00:04+00:00",
        })
        af = AgentFile(data)
        pairs = af.to_training_format(max_tool_output_len=100)
        # Find tool message in target
        for pair in pairs:
            for msg in pair["target"]:
                if msg["role"] == "tool":
                    assert len(msg["content"]) < 200


class TestProcessBatch:
    def test_processes_multiple_files(self, sample_af_data):
        # Create temp files
        paths = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".af", delete=False) as f:
                json.dump(sample_af_data, f)
                paths.append(Path(f.name))

        try:
            results = AgentFile.process_batch(paths, lambda af: af.num_messages)
            assert len(results) == 3
            for path, count in results:
                assert count == 6
        finally:
            for p in paths:
                p.unlink(missing_ok=True)

    def test_handles_errors(self, sample_af_data):
        # Create one valid and one invalid file
        paths = []
        with tempfile.NamedTemporaryFile(mode="w", suffix=".af", delete=False) as f:
            json.dump(sample_af_data, f)
            paths.append(Path(f.name))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".af", delete=False) as f:
            f.write("invalid json")
            paths.append(Path(f.name))

        try:
            results = AgentFile.process_batch(paths, lambda af: af.num_messages)
            assert len(results) == 2
            # One should succeed, one should fail
            successes = [r for r in results if r[1] is not None]
            failures = [r for r in results if r[1] is None]
            assert len(successes) == 1
            assert len(failures) == 1
        finally:
            for p in paths:
                p.unlink(missing_ok=True)


class TestWithRealFiles:
    """Tests using actual downloaded agentfiles (skip if not available)."""

    AGENT_DIR = Path(__file__).parent.parent / "extracted_contexts" / "raw"

    @pytest.fixture
    def real_af_path(self):
        if not self.AGENT_DIR.exists():
            pytest.skip("downloaded_agents_project_new not found")
        files = list(self.AGENT_DIR.glob("*.af"))
        if not files:
            pytest.skip("No .af files found")
        return files[0]

    def test_load_real_file(self, real_af_path):
        af = AgentFile.load(real_af_path)
        assert af.num_messages > 0
        assert af.name

    def test_revert_real_file(self, real_af_path):
        af = AgentFile.load(real_af_path)
        if af.num_messages <= 1:
            pytest.skip("Agent has no messages to revert")

        reverted = af.revert(1)
        assert reverted.num_messages < af.num_messages

    def test_roundtrip_real_file(self, real_af_path):
        af = AgentFile.load(real_af_path)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".af", delete=False) as f:
            path = Path(f.name)

        try:
            af.save(path)
            reloaded = AgentFile.load(path)
            assert reloaded.num_messages == af.num_messages
            assert reloaded.name == af.name
        finally:
            path.unlink(missing_ok=True)
