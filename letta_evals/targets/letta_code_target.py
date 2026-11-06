import asyncio
import json
import logging
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import anyio
from letta_client import AsyncLetta

from letta_evals.execution.trace import extract_usage_stats
from letta_evals.models import Sample, TargetResult, TargetTrace
from letta_evals.targets.errors import TargetError
from letta_evals.targets.local_backend import LocalAgentsClient, prompt_message, trajectory_message
from letta_evals.utils import load_object
from letta_evals.visualization.base import ProgressCallback

logger = logging.getLogger(__name__)

# Letta Code reads the agent's memory directory from these.
_MEMORY_DIR_VARS = ("MEMORY_DIR", "LETTA_MEMORY_DIR", "LETTA_MEMORY_DIR_EXPLICIT")


def _unpack_factory_result(result: Any) -> tuple[str, dict]:
    """Normalize an agent factory's return value to ``(agent_id, context)``."""
    if isinstance(result, str):
        return result, {}
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], str) and isinstance(result[1], dict):
        return result
    raise TypeError(f"Agent factory must return agent_id or (agent_id, context dict), got {type(result).__name__}")


def _rollout_vars(sample: Sample, context: Optional[dict]) -> dict:
    """``sample.extra_vars`` with the agent factory's per-rollout ``context`` layered on top.

    ``env`` merges per key so a factory can add variables without dropping
    dataset-provided ones.
    """
    extra_vars = sample.extra_vars or {}
    context = context or {}
    merged = {**extra_vars, **context}
    if isinstance(extra_vars.get("env"), dict) and isinstance(context.get("env"), dict):
        merged["env"] = {**extra_vars["env"], **context["env"]}
    return merged


def _resolve_memory_dir(sample: Sample, context: dict, cwd: Path) -> Optional[Path]:
    """Resolve the rollout's memory directory, or None when it has none.

    ``env["MEMORY_DIR"]`` (or ``env["LETTA_MEMORY_DIR"]``) from context layered
    over ``sample.extra_vars``. A relative path resolves against the working
    directory, as it would for Letta Code.
    """
    env = _rollout_vars(sample, context).get("env")
    memory_dir = (env.get("MEMORY_DIR") or env.get("LETTA_MEMORY_DIR")) if isinstance(env, dict) else None
    return (cwd / Path(str(memory_dir)).expanduser()).resolve() if memory_dir else None


@dataclass
class _Stream:
    """What an attempt has read from the CLI so far, kept so a failed attempt can still report it."""

    agent_id: Optional[str] = None
    # Last non-blank stdout line; the result event, with usage, once the CLI exits.
    last_line: str = ""
    # The trajectory as read from the stream; only kept on the local backend.
    messages: Optional[list] = None


class LettaCodeTarget:
    """Letta code target that invokes the letta CLI command."""

    def __init__(
        self,
        client: AsyncLetta,
        model_handle: str,
        default_cwd: Path,
        timeout: int = 300,
        max_retries: int = 0,
        base_url: Optional[str] = None,
        agent_script: Optional[str] = None,
        base_dir: Optional[Path] = None,
        flags: Optional[str] = None,
        permission_mode: Optional[str] = None,
        local_backend_dir: Optional[Path] = None,
        use_agent_model: bool = False,
    ):
        """Initialize the Letta Code target.

        Args:
            client: Letta client handed to the agent factory on the API backend
            model_handle: Model handle to use with letta code
            default_cwd: Working directory for the rollout when neither the
                factory's context nor ``sample.extra_vars`` sets ``cwd``. The
                runner creates a fresh empty directory per rollout.
            timeout: Command timeout in seconds (default: 300)
            max_retries: Number of retry attempts on failure
            base_url: Base URL for the Letta server (passed to CLI via env var)
            agent_script: Path to agent factory script (e.g., "setup_agent.py:setup_agent").
                If provided, the factory function is called to create an agent per sample,
                and --agent is used instead of --new-agent.
            base_dir: Suite directory; resolves relative ``agent_script`` and ``cwd`` paths
            flags: Additional CLI flags to pass to letta code, parsed with shell quoting
                rules (e.g., "--memfs --no-skills").
            permission_mode: Permission mode to pass through to Letta Code.
                Use a current Letta Code CLI mode such as "unrestricted",
                "standard", or "acceptEdits".
            local_backend_dir: If set, run Letta Code on its local backend
                (``letta --backend local``) with this directory as its store
                instead of against a Letta server. Agent factories get a
                ``LocalAgentsClient``, and the trajectory comes from the CLI's
                stream-json output.
            use_agent_model: If true, a factory-created agent runs on the model
                and model settings its factory gave it: no ``--model``, and on the
                local backend the factory's model isn't replaced at creation.
                ``model_handle`` then only labels results.

        Agent factories may return ``(agent_id, context)``. ``context["env"]``
        (dict) and ``context["cwd"]`` configure the subprocess for that rollout,
        layered over the same keys in ``sample.extra_vars``; user env keys win
        over target-managed ones. ``env["MEMORY_DIR"]`` gives the agent's memory
        directory. See ``_build_subprocess_env``. The working directory is
        written back to ``context["cwd"]`` so extractors can read it.

        Choosing a working directory changes where the agent starts and how
        relative paths resolve; it does not confine the agent. Configure
        ``suite.sandbox`` for filesystem isolation.
        """
        self.client = client
        self.model_handle = model_handle
        self.permission_mode = permission_mode
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url
        self.agent_script = agent_script
        # base_dir is the suite directory (set by SuiteSpec.from_yaml). It
        # resolves agent_script paths and relative `cwd` values.
        self.base_dir = base_dir or Path.cwd()
        self.default_cwd = default_cwd
        self.flags = shlex.split(flags) if flags else []
        self.local_backend_dir = local_backend_dir
        self.use_agent_model = use_agent_model

    def _resolve_cwd(self, sample: Sample, agent_id: Optional[str], context: Optional[dict] = None) -> Path:
        """Resolve the subprocess working directory.

        ``cwd`` from context layered over ``sample.extra_vars``, resolved
        against the suite directory when relative; otherwise ``default_cwd``.
        An explicit ``cwd`` must already exist.
        """
        cwd = _rollout_vars(sample, context).get("cwd")
        if not cwd:
            return self.default_cwd.resolve()
        path = (self.base_dir / Path(str(cwd)).expanduser()).resolve()
        if not path.is_dir():
            raise TargetError(f"cwd {path} for sample {sample.id} is not a directory", agent_id=agent_id)
        return path

    def _build_subprocess_env(
        self,
        sample: Sample,
        agent_id: Optional[str],
        context: Optional[dict] = None,
        memory_dir: Optional[Path] = None,
    ) -> dict[str, str]:
        """Build subprocess env: os.environ -> target-managed -> extra_vars/context ``env`` -> memory directory."""
        env = os.environ.copy()
        # A rollout's memory directory comes only from its own env, never from the
        # environment letta-evals runs in (such as a Letta Code session's own).
        for key in _MEMORY_DIR_VARS:
            env.pop(key, None)
        # A rollout is never a session to resume. Without this, Letta Code records
        # each run as the last session of the working directory and of the user.
        env["LETTA_DISABLE_SESSION_PERSIST"] = "1"

        if self.local_backend_dir:
            env["LETTA_LOCAL_BACKEND_DIR"] = str(self.local_backend_dir)
        elif self.base_url:
            env["LETTA_BASE_URL"] = self.base_url
            logger.info(f"Setting LETTA_BASE_URL={self.base_url} for letta CLI")

        sample_env = _rollout_vars(sample, context).get("env")
        if sample_env is not None:
            if not isinstance(sample_env, dict):
                raise TargetError(
                    f"'env' in extra_vars/context must be a dict, got {type(sample_env).__name__} for sample {sample.id}",
                    agent_id=agent_id,
                )
            applied = []
            for k, v in sample_env.items():
                if not isinstance(k, str):
                    raise TargetError(
                        f"'env' keys must be strings, got {type(k).__name__} ({k!r}) for sample {sample.id}",
                        agent_id=agent_id,
                    )
                # Coerce to str — env values must be strings; None becomes empty.
                env[k] = "" if v is None else str(v)
                applied.append(k)
            if applied:
                logger.info(f"Applied per-sample env overrides for sample {sample.id}: keys={sorted(applied)}")

        if memory_dir:
            # The resolved path replaces a relative one. Without the marker, Letta Code
            # drops a MEMORY_DIR outside its memory store from the tool shells of agents
            # without MemFS (e.g. --stateless).
            env["MEMORY_DIR"] = env["LETTA_MEMORY_DIR"] = str(memory_dir)
            env["LETTA_MEMORY_DIR_EXPLICIT"] = "1"

        return env

    async def run(
        self,
        sample: Sample,
        progress_callback: Optional[ProgressCallback] = None,
        project_id: Optional[str] = None,
    ) -> TargetResult:
        """Run the letta CLI command on a sample and return execution metadata.

        A failed attempt is retried up to ``max_retries`` times with exponential
        backoff; the last attempt's ``TargetError`` is raised.

        The target deliberately does not fetch server-side trace fields such as
        messages, agent state, or token data. ``Runner`` owns those fetches so
        in-process and sandboxed runs share the same trace assembly path. On the
        local backend there is no server, so the target returns a ``TargetTrace``
        whose trajectory it read from the stream; token data stays unset.
        """
        for attempt in range(self.max_retries):
            try:
                return await self._run_once(sample, progress_callback)
            except TargetError as e:
                backoff_time = 2**attempt
                logger.warning(
                    f"Letta command failed for sample {sample.id} (attempt {attempt + 1}/{self.max_retries + 1}). "
                    f"Agent: {e.agent_id or 'unknown'}. "
                    f"Error: {type(e.__cause__).__name__}: {e.__cause__}. Retrying in {backoff_time}s..."
                )
                await anyio.sleep(backoff_time)
        return await self._run_once(sample, progress_callback)

    async def _run_once(self, sample: Sample, progress_callback: Optional[ProgressCallback]) -> TargetResult:
        """Run one attempt: create the agent, run the CLI on the sample's prompt and collect the result.

        Any failure is raised as a ``TargetError`` carrying what the attempt
        learned first: the agent id, usage so far and, on the local backend, the
        partial trajectory. Runner fetches any server-side trace fields from the
        agent id.
        """
        stream = _Stream()
        try:
            factory_agent_id, context = await self._create_agent(sample, progress_callback)
            stream.agent_id = factory_agent_id

            # The working directory is handed forward so extractors don't re-derive it.
            cwd = self._resolve_cwd(sample, factory_agent_id, context)
            context = {**context, "cwd": str(cwd)}
            memory_dir = _resolve_memory_dir(sample, context, cwd)
            if memory_dir:
                memory_dir.mkdir(parents=True, exist_ok=True)

            # Built after the factory call, which may modify the sample. Multiple
            # inputs are sent as one prompt, joined by newlines.
            inputs = sample.input if isinstance(sample.input, list) else [sample.input]
            prompt = "\n".join(str(inp) for inp in inputs).replace("{pwd}", cwd.as_posix())

            env = self._build_subprocess_env(sample, factory_agent_id, context, memory_dir)
            await self._run_cli(sample, prompt, self._command(factory_agent_id), cwd, env, stream, progress_callback)

            if not stream.agent_id:
                raise RuntimeError("No agent_id found in letta stream output")
            usage_stats = extract_usage_stats(stream.last_line)
            if stream.messages is not None:
                return TargetTrace(
                    agent_id=stream.agent_id,
                    model_handle=self.model_handle,
                    agent_usage=usage_stats,
                    context=context,
                    trajectory=[stream.messages],
                )
            return TargetResult(
                agent_id=stream.agent_id,
                model_handle=self.model_handle,
                agent_usage=usage_stats,
                context=context,
            )
        except Exception as e:
            timeout_hint = f"Timed out after {self.timeout}s" if isinstance(e, TimeoutError) else ""
            raise TargetError(
                str(e) or timeout_hint or type(e).__name__,
                agent_id=stream.agent_id,
                agent_usage=extract_usage_stats(stream.last_line),
                trajectory=[stream.messages] if stream.messages else None,
            ) from e

    async def _create_agent(
        self, sample: Sample, progress_callback: Optional[ProgressCallback]
    ) -> tuple[Optional[str], dict]:
        """Create the sample's agent with the agent factory: ``(agent_id, context)``.

        Without a factory this is ``(None, {})``, and the CLI creates a new agent.
        """
        if not self.agent_script:
            return None, {}
        agent_factory_func = load_object(self.agent_script, self.base_dir)
        client = (
            LocalAgentsClient(
                self.local_backend_dir,
                self._build_subprocess_env(sample, None),
                None if self.use_agent_model else self.model_handle,
                self.timeout,
            )
            if self.local_backend_dir
            else self.client
        )
        agent_id, context = _unpack_factory_result(await agent_factory_func(client, sample))
        logger.info(f"Created agent {agent_id} via agent_factory for sample {sample.id}")
        if progress_callback:
            await progress_callback.agent_created(sample.id, agent_id=agent_id, model_handle=self.model_handle)
        return agent_id, context

    def _command(self, factory_agent_id: Optional[str]) -> list[str]:
        """The letta CLI command: headless, with stream-json output and the prompt on stdin."""
        cmd = ["letta"]
        if self.local_backend_dir:
            cmd.extend(["--backend", "local"])
        cmd.extend(["--output-format", "stream-json"])
        # --model on resume applies the model catalog's settings over the agent's
        # own. A local factory agent was already created with this model, and
        # with use_agent_model the factory owns the model and its settings.
        if not (factory_agent_id and (self.local_backend_dir or self.use_agent_model)):
            cmd.extend(["--model", self.model_handle])
        # Only use --yolo when no explicit permission_mode is set,
        # since --yolo overrides --permission-mode (sets bypassPermissions).
        if not self.permission_mode:
            cmd.append("--yolo")
        cmd.extend(["--agent", factory_agent_id] if factory_agent_id else ["--new-agent"])
        cmd.extend(self.flags)
        if self.permission_mode:
            cmd.extend(["--permission-mode", self.permission_mode])
        # With no positional prompt, -p reads it from stdin, which avoids the OS's ARG_MAX limit.
        cmd.append("-p")
        return cmd

    async def _run_cli(
        self,
        sample: Sample,
        prompt: str,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        stream: _Stream,
        progress_callback: Optional[ProgressCallback],
    ) -> None:
        """Run the CLI with the prompt on stdin, recording its stream-json output in ``stream``.

        Raises when the CLI times out or exits non-zero.
        """
        prompt_bytes = prompt.encode("utf-8")
        logger.info(f"Running letta command for sample {sample.id} (prompt via stdin, {len(prompt_bytes)} bytes)")
        if self.local_backend_dir:
            stream.messages = [prompt_message(prompt)]
        num_inputs = len(sample.input) if isinstance(sample.input, list) else 1
        stderr_chunks = []

        # The StreamReader limit is well above asyncio's 64 KiB default: one
        # stream-json event per line, and a single event (a Read of a large
        # file, verbose Bash output, a base64-encoded image) can exceed 64 KiB.
        # Without this, reading stdout raises LimitOverrunError on long tool
        # results and crashes the rollout.
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=env,
            limit=16 * 1024 * 1024,
        )
        process.stdin.write(prompt_bytes)
        await process.stdin.drain()
        process.stdin.close()
        await process.stdin.wait_closed()

        # Only the init event (agent_id) and the final result event (usage,
        # parsed from last_line after exit) are needed from the stream when the
        # trajectory is fetched server-side, so everything after init is drained
        # without parsing (#319). On the local backend the rest of the stream is
        # the trajectory.
        async def _read_stdout():
            init_seen = False
            run_id = None
            async for raw_line in process.stdout:
                line = raw_line.decode(errors="replace").strip()
                if not line:
                    continue
                stream.last_line = line
                if init_seen and stream.messages is None:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                if init_seen:
                    run_id = event.get("run_id") or run_id
                    if (message := trajectory_message(event, run_id)) is not None:
                        stream.messages.append(message)
                    continue
                if event.get("type") == "system" and event.get("subtype") == "init":
                    init_seen = True
                    if not stream.agent_id:
                        stream.agent_id = event.get("agent_id")
                        logger.info(f"Captured agent_id {stream.agent_id} from stream init event")
                        if progress_callback and stream.agent_id:
                            await progress_callback.agent_created(
                                sample.id, agent_id=stream.agent_id, model_handle=self.model_handle
                            )
                    if progress_callback and stream.agent_id:
                        await progress_callback.message_sending(
                            sample.id, 1, num_inputs, agent_id=stream.agent_id, model_handle=self.model_handle
                        )

        async def _read_stderr():
            async for raw_line in process.stderr:
                stderr_chunks.append(raw_line.decode())

        try:
            await asyncio.wait_for(
                asyncio.gather(_read_stdout(), _read_stderr(), process.wait()),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(f"Letta command timed out after {self.timeout} seconds")

        if process.returncode != 0:
            # Surface the last stdout line so the real error isn't lost
            parts = [f"Letta command failed with return code {process.returncode}"]
            if stream.last_line:
                parts.append(f"Last stdout line: {stream.last_line[:500]}")
            stderr_text = "".join(stderr_chunks)
            if stderr_text:
                parts.append(f"Stderr: {stderr_text[:500]}")
            raise RuntimeError(". ".join(parts))
