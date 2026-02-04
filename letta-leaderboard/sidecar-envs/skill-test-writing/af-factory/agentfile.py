"""AgentFile wrapper for manipulating Letta .af files.

Immutable operations:
- AgentFile.load(path) - load from disk
- AgentFile.create_empty() - blank agent
- af.revert(n) - remove last N messages
- af.run_query(prompt, client) - send message, get response
- af.diff(other) - show differences
- af.save(path) - write to disk
"""

from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from letta_client import AsyncLetta


@dataclass
class AgentFile:
    """Immutable wrapper for .af files."""

    data: dict = field(repr=False)

    @classmethod
    def load(cls, path: Path | str) -> AgentFile:
        """Load an .af file from disk."""
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def create_empty(
        cls,
        name: str = "empty-agent",
        model_handle: str = "openai/gpt-4o-mini",
        system_prompt: str | None = None,
    ) -> AgentFile:
        """Create a blank agent with minimal configuration."""
        now = datetime.now(timezone.utc).isoformat()
        system = system_prompt or "You are a helpful assistant."

        data = {
            "agents": [
                {
                    "name": name,
                    "memory_blocks": [],
                    "tools": [],
                    "tool_ids": [],
                    "source_ids": [],
                    "block_ids": [],
                    "tool_rules": [],
                    "tags": [],
                    "system": system,
                    "agent_type": "letta_v1_agent",
                    "llm_config": {
                        "model": model_handle.split("/")[-1],
                        "model_endpoint_type": model_handle.split("/")[0],
                        "handle": model_handle,
                        "context_window": 128000,
                        "temperature": 0.7,
                    },
                    "embedding_config": {
                        "embedding_endpoint_type": "openai",
                        "embedding_model": "text-embedding-3-small",
                        "embedding_dim": 1536,
                        "handle": "openai/text-embedding-3-small",
                    },
                    "include_base_tools": False,
                    "id": "agent-0",
                    "in_context_message_ids": ["message-0"],
                    "messages": [
                        {
                            "type": "message",
                            "role": "system",
                            "content": [{"type": "text", "text": system}],
                            "id": "message-0",
                            "agent_id": "agent-0",
                            "tool_calls": None,
                            "tool_returns": [],
                            "created_at": now,
                        }
                    ],
                }
            ],
            "groups": [],
            "blocks": [],
            "files": [],
            "sources": [],
            "tools": [],
            "mcp_servers": [],
            "metadata": {"revision_id": "empty"},
            "created_at": now,
        }
        return cls(data)

    def save(self, path: Path | str) -> None:
        """Save to disk."""
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)

    def revert(self, n_steps: int = 1) -> AgentFile:
        """Return new AgentFile with last N messages removed.

        A "step" is a user message + all subsequent assistant/tool messages until the next user message.
        To simply remove N raw messages, use revert_messages().
        """
        data = copy.deepcopy(self.data)
        agent = data["agents"][0]
        messages = agent["messages"]

        # Find user message indices (excluding system)
        user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]

        if n_steps >= len(user_indices):
            # Keep only system message
            cutoff = 1
        else:
            # Keep up to the nth-from-last user message
            cutoff = user_indices[-n_steps]

        agent["messages"] = messages[:cutoff]
        agent["in_context_message_ids"] = [m["id"] for m in agent["messages"]]

        return AgentFile(data)

    def revert_messages(self, n: int = 1) -> AgentFile:
        """Return new AgentFile with last N raw messages removed."""
        data = copy.deepcopy(self.data)
        agent = data["agents"][0]

        # Keep at least system message
        keep_count = max(1, len(agent["messages"]) - n)
        agent["messages"] = agent["messages"][:keep_count]
        agent["in_context_message_ids"] = [m["id"] for m in agent["messages"]]

        return AgentFile(data)

    async def run_query(
        self,
        prompt: str,
        client: AsyncLetta,
        model_handle: str | None = None,
    ) -> AgentFile:
        """Send message and return new AgentFile with response appended.

        Imports agent to server, sends message, exports result.
        Original AgentFile is unchanged (immutable).
        """
        # Write to temp file for import
        with tempfile.NamedTemporaryFile(mode="w", suffix=".af", delete=False) as f:
            json.dump(self.data, f)
            temp_path = f.name

        try:
            # Import agent
            with open(temp_path, "rb") as f:
                resp = await client.agents.import_file(
                    file=f,
                    append_copy_suffix=False,
                    override_existing_tools=False,
                    model=model_handle,
                )
            agent_id = resp.agent_ids[0]

            try:
                # Send message
                from letta_client.types import MessageCreateParam

                response = await client.agents.messages.create(
                    agent_id=agent_id,
                    messages=[MessageCreateParam(role="user", content=prompt)],
                )

                # Export updated agent
                exported = await client.agents.export_file(agent_id=agent_id)
                new_data = json.loads(exported)

                return AgentFile(new_data)

            finally:
                # Cleanup: delete the temporary agent
                await client.agents.delete(agent_id=agent_id)

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def diff(self, other: AgentFile) -> str:
        """Compare two AgentFiles and return human-readable diff."""
        lines = []

        # Compare message counts
        self_msgs = self.messages
        other_msgs = other.messages
        lines.append(f"Messages: {len(self_msgs)} -> {len(other_msgs)}")

        # Find new messages
        self_ids = {m["id"] for m in self_msgs}
        new_msgs = [m for m in other_msgs if m["id"] not in self_ids]

        if new_msgs:
            lines.append(f"\nNew messages ({len(new_msgs)}):")
            for m in new_msgs:
                role = m["role"]
                content = self._extract_content(m)
                preview = content[:100] + "..." if len(content) > 100 else content
                lines.append(f"  [{role}] {preview}")

        # Compare blocks
        self_blocks = {b.get("label"): b for b in self.data.get("blocks", [])}
        other_blocks = {b.get("label"): b for b in other.data.get("blocks", [])}

        new_block_labels = set(other_blocks) - set(self_blocks)
        if new_block_labels:
            lines.append(f"\nNew blocks: {new_block_labels}")

        changed_blocks = []
        for label in set(self_blocks) & set(other_blocks):
            if self_blocks[label].get("value") != other_blocks[label].get("value"):
                changed_blocks.append(label)
        if changed_blocks:
            lines.append(f"Changed blocks: {changed_blocks}")

        return "\n".join(lines)

    def _extract_content(self, message: dict) -> str:
        """Extract text content from a message."""
        content = message.get("content", [])
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict)]
            return " ".join(texts)
        return str(content)

    @property
    def agent(self) -> dict:
        """Get first agent dict."""
        return self.data["agents"][0]

    @property
    def messages(self) -> list[dict]:
        """Get messages from first agent."""
        return self.agent["messages"]

    @property
    def num_messages(self) -> int:
        """Number of messages."""
        return len(self.messages)

    @property
    def name(self) -> str:
        """Agent name."""
        return self.agent.get("name", "unnamed")

    @property
    def model(self) -> str:
        """Model handle."""
        llm_config = self.agent.get("llm_config", {})
        return llm_config.get("handle") or llm_config.get("model", "unknown")

    def get_conversation(self, include_system: bool = False) -> list[dict]:
        """Get conversation as list of {role, content} dicts."""
        result = []
        for m in self.messages:
            if m["role"] == "system" and not include_system:
                continue
            result.append({
                "role": m["role"],
                "content": self._extract_content(m),
                "tool_calls": m.get("tool_calls"),
            })
        return result

    def __repr__(self) -> str:
        return f"AgentFile(name={self.name!r}, messages={self.num_messages}, model={self.model!r})"

    def min_turns(self, n: int) -> bool:
        """Check if agent has at least n user turns."""
        user_turns = sum(1 for m in self.messages if m["role"] == "user")
        return user_turns >= n

    def truncate_tool_outputs(self, max_len: int = 500) -> AgentFile:
        """Return new AgentFile with tool outputs truncated to max_len."""
        data = copy.deepcopy(self.data)

        for agent in data.get("agents", []):
            for msg in agent.get("messages", []):
                if msg.get("role") != "tool":
                    continue

                content = msg.get("content", [])
                if isinstance(content, str):
                    if len(content) > max_len:
                        msg["content"] = content[:max_len] + "... [truncated]"
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            text = item["text"]
                            if len(text) > max_len:
                                item["text"] = text[:max_len] + "... [truncated]"

        return AgentFile(data)

    @classmethod
    def process_batch(
        cls,
        paths: list[Path | str],
        fn: callable,
        max_workers: int | None = None,
    ) -> list:
        """Process multiple agentfiles in parallel.

        Args:
            paths: List of .af file paths
            fn: Function taking AgentFile, returning any result
            max_workers: Max parallel workers (default: CPU count)

        Returns:
            List of (path, result) tuples. Failed items have result=None.

        Example:
            results = AgentFile.process_batch(
                paths,
                lambda af: af.strip_system_reminders().to_training_format()
            )
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing

        if max_workers is None:
            max_workers = multiprocessing.cpu_count()

        results = []

        # Use ThreadPoolExecutor for I/O bound work (file loading)
        from concurrent.futures import ThreadPoolExecutor

        def process_one(path):
            try:
                af = cls.load(path)
                return (path, fn(af))
            except Exception as e:
                return (path, None)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_one, p): p for p in paths}
            for future in as_completed(futures):
                results.append(future.result())

        return results

    def randomize_identity(self, seed: int | None = None) -> AgentFile:
        """Randomize agent/message IDs and timestamps for anonymization.

        Replaces:
        - agent IDs (agent-xxxx)
        - message IDs (message-xxxx)
        - tool call IDs (call_xxxx)
        - timestamps
        - agent name
        """
        import random

        rng = random.Random(seed)
        data = copy.deepcopy(self.data)

        def random_id(prefix: str) -> str:
            return f"{prefix}-{rng.randbytes(6).hex()}"

        def random_call_id() -> str:
            return f"call_{rng.randbytes(12).hex()}"

        # Build ID mapping
        agent_id_map = {}
        message_id_map = {}
        tool_call_id_map = {}

        # Map agent IDs
        for agent in data.get("agents", []):
            old_id = agent.get("id", "")
            if old_id:
                agent_id_map[old_id] = random_id("agent")

        # Map message and tool call IDs
        for agent in data.get("agents", []):
            for msg in agent.get("messages", []):
                old_id = msg.get("id", "")
                if old_id:
                    message_id_map[old_id] = random_id("message")

                # Map tool call IDs
                for tc in msg.get("tool_calls") or []:
                    old_tc_id = tc.get("id", "")
                    if old_tc_id:
                        tool_call_id_map[old_tc_id] = random_call_id()

        # Apply mappings
        base_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        for agent in data.get("agents", []):
            # Randomize agent
            agent["id"] = agent_id_map.get(agent.get("id", ""), random_id("agent"))
            agent["name"] = f"agent-{rng.randint(1000, 9999)}"

            # Update in_context_message_ids
            agent["in_context_message_ids"] = [
                message_id_map.get(mid, mid) for mid in agent.get("in_context_message_ids", [])
            ]

            # Update messages
            for i, msg in enumerate(agent.get("messages", [])):
                msg["id"] = message_id_map.get(msg.get("id", ""), random_id("message"))
                msg["agent_id"] = agent["id"]
                msg["created_at"] = (base_time + timedelta(seconds=i * 10)).isoformat()

                # Update tool call IDs
                for tc in msg.get("tool_calls") or []:
                    tc["id"] = tool_call_id_map.get(tc.get("id", ""), random_call_id())

                # Update tool_call_id references
                if msg.get("tool_call_id"):
                    msg["tool_call_id"] = tool_call_id_map.get(msg["tool_call_id"], msg["tool_call_id"])

        # Randomize top-level metadata
        data["created_at"] = base_time.isoformat()
        data["metadata"] = {"revision_id": rng.randbytes(6).hex()}

        return AgentFile(data)

    def strip_system_reminders(self) -> AgentFile:
        """Remove messages that are system reminders (automated context messages).

        Removes user messages starting with '<system-reminder>' or containing
        'MEMORY CHECK' automated prompts.
        """
        data = copy.deepcopy(self.data)

        for agent in data.get("agents", []):
            filtered_messages = []
            skip_next_tool_responses = set()  # Track tool calls to skip their responses

            for msg in agent.get("messages", []):
                content = self._extract_content(msg)
                role = msg.get("role", "")

                # Check if this is a system reminder
                is_system_reminder = False
                if role == "user":
                    if content.strip().startswith("<system-reminder>"):
                        is_system_reminder = True
                    elif "MEMORY CHECK:" in content:
                        is_system_reminder = True
                    elif content.strip().startswith("{") and '"type": "system_alert"' in content:
                        is_system_reminder = True

                if is_system_reminder:
                    # Mark any tool calls from the next assistant message to skip
                    continue

                filtered_messages.append(msg)

            agent["messages"] = filtered_messages
            agent["in_context_message_ids"] = [m["id"] for m in filtered_messages]

        return AgentFile(data)

    def to_training_format(
        self,
        include_system: bool = True,
        max_tool_output_len: int = 1000,
        source: str | None = None,
    ) -> list[dict]:
        """Convert to training format: list of (context, target) pairs.

        Each user turn becomes a training example where:
        - context: all messages up to and including the user message
        - target: assistant response(s) until next user message

        Returns list of dicts with keys:
        - id: unique identifier
        - context: list of {role, content, ?tool_calls} messages
        - target: list of {role, content, ?tool_calls, ?name} messages
        - metadata: {source, turn_index, tools_used}
        """
        pairs = []
        messages = self.messages

        # Find user message indices (excluding system)
        user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]

        for turn_idx, user_msg_idx in enumerate(user_indices):
            # Context: everything up to and including this user message
            context_msgs = messages[: user_msg_idx + 1]

            # Target: everything after user message until next user message (or end)
            if turn_idx + 1 < len(user_indices):
                next_user_idx = user_indices[turn_idx + 1]
                target_msgs = messages[user_msg_idx + 1 : next_user_idx]
            else:
                target_msgs = messages[user_msg_idx + 1 :]

            # Skip if no target (last user message with no response)
            if not target_msgs:
                continue

            # Format context
            context = []
            for m in context_msgs:
                entry = {"role": m["role"], "content": self._extract_content(m)}
                if m.get("tool_calls"):
                    entry["tool_calls"] = [
                        {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                        for tc in m["tool_calls"]
                    ]
                if not include_system and m["role"] == "system":
                    continue
                context.append(entry)

            # Format target
            target = []
            tools_used = set()
            for m in target_msgs:
                content = self._extract_content(m)
                # Truncate tool outputs
                if m["role"] == "tool" and len(content) > max_tool_output_len:
                    content = content[:max_tool_output_len] + "... [truncated]"

                entry = {"role": m["role"], "content": content}
                if m.get("tool_calls"):
                    entry["tool_calls"] = [
                        {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                        for tc in m["tool_calls"]
                    ]
                    tools_used.update(tc["function"]["name"] for tc in m["tool_calls"])
                if m.get("name"):
                    entry["name"] = m["name"]
                target.append(entry)

            pairs.append(
                {
                    "id": f"{source or self.name}_turn_{turn_idx}",
                    "context": context,
                    "target": target,
                    "metadata": {
                        "source": source or self.name,
                        "turn_index": turn_idx,
                        "tools_used": sorted(tools_used),
                        "context_messages": len(context),
                        "target_messages": len(target),
                    },
                }
            )

        return pairs


# CLI interface
if __name__ == "__main__":
    import sys

    def main():
        if len(sys.argv) < 2:
            print("Usage: python agentfile.py <command> [args]")
            print("Commands:")
            print("  info <file.af>           - Show agent info")
            print("  messages <file.af>       - List messages")
            print("  revert <file.af> <n>     - Revert N steps, print to stdout")
            print("  diff <file1.af> <file2.af> - Compare two files")
            print("  create <name> [model]    - Create empty agent, print to stdout")
            return

        cmd = sys.argv[1]

        if cmd == "info":
            af = AgentFile.load(sys.argv[2])
            print(af)
            print(f"System prompt: {len(af.agent.get('system', ''))} chars")
            print(f"Tools: {len(af.agent.get('tool_ids', []))}")
            print(f"Blocks: {len(af.data.get('blocks', []))}")

        elif cmd == "messages":
            af = AgentFile.load(sys.argv[2])
            for i, m in enumerate(af.messages):
                role = m["role"]
                content = af._extract_content(m)[:80]
                print(f"{i}: [{role}] {content}")

        elif cmd == "revert":
            af = AgentFile.load(sys.argv[2])
            n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
            reverted = af.revert(n)
            print(json.dumps(reverted.data, indent=2))

        elif cmd == "diff":
            af1 = AgentFile.load(sys.argv[2])
            af2 = AgentFile.load(sys.argv[3])
            print(af1.diff(af2))

        elif cmd == "create":
            name = sys.argv[2] if len(sys.argv) > 2 else "empty-agent"
            model = sys.argv[3] if len(sys.argv) > 3 else "openai/gpt-4o-mini"
            af = AgentFile.create_empty(name=name, model_handle=model)
            print(json.dumps(af.data, indent=2))

        else:
            print(f"Unknown command: {cmd}")

    main()
