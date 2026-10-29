"""Helpers for running Letta Code on its local backend (``backend: local``).

The local backend keeps agents and conversations on disk instead of on a Letta
server, so agent factories run against ``LocalAgentsClient`` (which creates the
agent with ``letta agents create --body``) and the trajectory is read from the
CLI's stream-json output.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from letta_client.types import AgentState
from pydantic import TypeAdapter, ValidationError

from letta_evals.models import LettaMessageUnion

logger = logging.getLogger(__name__)

_MESSAGE_ADAPTER: TypeAdapter = TypeAdapter(LettaMessageUnion)
# Stream event fields that are not part of the API message schema.
_STREAM_ONLY_FIELDS = ("type", "session_id", "uuid", "timestamp")
# Message types the stream emits besides trajectory messages.
_NON_TRAJECTORY_TYPES = frozenset({"usage_statistics", "stop_reason"})
# Stream names for message types that some letta-client versions name differently.
_SDK_MESSAGE_TYPE_ALIASES = {"summary_message": "summary", "event_message": "event"}


def local_memory_dir(storage_dir: Path, agent_id: str) -> Path:
    """Memory root of a local-backend agent with memfs enabled."""
    return storage_dir / "memfs" / agent_id / "memory"


def trajectory_message(event: dict, run_id: Optional[str]) -> Optional[Any]:
    """Convert one stream-json ``message`` event into a trajectory message, or None."""
    if event.get("type") != "message" or event.get("message_type") in _NON_TRAJECTORY_TYPES:
        return None
    data = {key: value for key, value in event.items() if key not in _STREAM_ONLY_FIELDS}
    # The server groups one model call's messages by step_id; locally each model
    # call is its own run, and tool events carry no run_id of their own.
    if run_id and not data.get("step_id"):
        data["step_id"] = f"step-{run_id}"
    try:
        return _MESSAGE_ADAPTER.validate_python(data)
    except ValidationError as error:
        alias = _SDK_MESSAGE_TYPE_ALIASES.get(data["message_type"])
        if alias is not None:
            try:
                return _MESSAGE_ADAPTER.validate_python({**data, "message_type": alias})
            except ValidationError:
                pass
        logger.warning("Dropping a %s the message schema rejects: %s", data.get("message_type"), error)
        return None


class _LocalAgents:
    def __init__(self, storage_dir: Path, env: dict[str, str], model_handle: str, timeout: float):
        self._storage_dir = storage_dir
        self._env = env
        self._model_handle = model_handle
        self._timeout = timeout

    async def create(self, **body: Any) -> AgentState:
        # The suite's model handle wins, as --model does for server-backed agents.
        # parallel_tool_calls defaults on as in Letta Code's own agents; without it,
        # the first resume rewrites the agent's model settings from its model preset.
        model_settings = {"parallel_tool_calls": True, **(body.get("model_settings") or {})}
        body = {**body, "model": self._model_handle, "model_settings": model_settings}
        process = await asyncio.create_subprocess_exec(
            "letta",
            "--backend",
            "local",
            "agents",
            "create",
            "--body",
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._storage_dir),
            env=self._env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(json.dumps(body, default=str).encode()), timeout=self._timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(f"letta agents create timed out after {self._timeout}s")
        if process.returncode != 0:
            detail = stderr.decode(errors="replace").strip() or stdout.decode(errors="replace").strip()
            raise RuntimeError(f"letta agents create failed with return code {process.returncode}: {detail[:500]}")
        # The local projection carries fields some SDK versions don't model, so skip validation.
        return AgentState.model_construct(**json.loads(stdout))

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(_unsupported(f"agents.{name}"))


def _unsupported(name: str) -> str:
    return f"Agent factories on the local backend can only call client.agents.create (got client.{name})"


class LocalAgentsClient:
    """The client agent factories receive on the local backend.

    Only ``agents.create`` is available: the agent is created in this sample's
    local store, and anything else a factory would do on a server has no local
    equivalent.
    """

    def __init__(self, storage_dir: Path, env: dict[str, str], model_handle: str, timeout: float):
        self.agents = _LocalAgents(storage_dir, env, model_handle, timeout)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(_unsupported(name))
