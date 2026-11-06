import asyncio
import json
import logging
import os
import shlex
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
            client: AsyncLetta client for retrieving messages after CLI execution
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
    ) -> TargetResult:
        """Run the letta CLI command on a sample and return execution metadata.

        The target deliberately does not fetch server-side trace fields such as
        messages, agent state, or token data. ``Runner`` owns those fetches so
        in-process and sandboxed runs share the same trace assembly path. On the
        local backend there is no server, so the target returns a ``TargetTrace``
        whose trajectory it read from the stream; token data stays unset.
        """
        attempt = 0
        last_error = None

        while attempt <= self.max_retries:
            # The trajectory as read from the stream; only kept on the local backend.
            messages: Optional[list] = None
            try:
                agent_id = None
                # Last non-blank stdout line; parsed after exit for the result event.
                last_line = ""

                # If agent_script is provided, create agent via factory first
                factory_agent_id = None
                context: dict = {}
                if self.agent_script:
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
                    factory_agent_id, context = _unpack_factory_result(await agent_factory_func(client, sample))
                    logger.info(f"Created agent {factory_agent_id} via agent_factory for sample {sample.id}")
                    if progress_callback:
                        await progress_callback.agent_created(
                            sample.id, agent_id=factory_agent_id, model_handle=self.model_handle
                        )

                # The working directory is handed forward so extractors don't re-derive it.
                cwd = self._resolve_cwd(sample, factory_agent_id, context)
                context = {**context, "cwd": str(cwd)}
                memory_dir = _resolve_memory_dir(sample, context, cwd)
                if memory_dir:
                    memory_dir.mkdir(parents=True, exist_ok=True)

                # construct prompt after factory call so agent_factory can modify sample if needed
                # handle single or multiple inputs
                inputs = sample.input if isinstance(sample.input, list) else [sample.input]

                # for multiple inputs, concatenate with newlines
                prompt = "\n".join(str(inp) for inp in inputs)
                prompt = prompt.replace("{pwd}", cwd.as_posix())

                # construct the letta-code CLI command (headless streaming JSON output).
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

                if factory_agent_id:
                    cmd.extend(["--agent", factory_agent_id])
                else:
                    cmd.append("--new-agent")

                # append any extra flags from suite config
                if self.flags:
                    cmd.extend(self.flags)

                if self.permission_mode:
                    cmd.extend(["--permission-mode", self.permission_mode])

                # Pass prompt via stdin to avoid OS ARG_MAX limits on large inputs.
                # letta-code headless mode reads from stdin when no positional prompt given.
                cmd.append("-p")
                prompt_bytes = prompt.encode("utf-8")

                logger.info(
                    f"Running letta command for sample {sample.id} (prompt via stdin, {len(prompt_bytes)} bytes)"
                )

                # Prepare environment variables for the subprocess.
                # Layers target-managed env (LETTA_BASE_URL) and per-rollout overrides
                # from extra_vars/context "env" (see class docstring).
                env = self._build_subprocess_env(sample, factory_agent_id, context, memory_dir)
                if self.local_backend_dir:
                    messages = [prompt_message(prompt)]

                agent_id = factory_agent_id
                stderr_chunks = []

                # run the letta command with prompt piped via stdin
                #
                # NOTE: We bump the StreamReader limit well above asyncio's
                # 64 KiB default. letta-code emits one JSON event per line in
                # stream-json mode, and a single event (e.g. a Read of a large
                # file, verbose Bash output, or a base64-encoded image) can
                # easily exceed 64 KiB. Without this, `async for raw_line in
                # process.stdout` below raises LimitOverrunError on long-tool-
                # result turns and crashes the rollout.
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(cwd),
                    env=env,
                    limit=16 * 1024 * 1024,
                )
                # Write prompt to stdin and close it
                process.stdin.write(prompt_bytes)
                await process.stdin.drain()
                process.stdin.close()
                await process.stdin.wait_closed()

                # Only the init event (agent_id) and the final result event
                # (usage, parsed from last_line after exit) are needed from the
                # stream when the trajectory is fetched server-side, so
                # everything after init is drained without parsing (#319). On
                # the local backend the rest of the stream is the trajectory.
                async def _read_stdout():
                    nonlocal agent_id, last_line
                    init_seen = False
                    run_id = None
                    async for raw_line in process.stdout:
                        line = raw_line.decode(errors="replace").strip()
                        if not line:
                            continue
                        last_line = line
                        if init_seen and messages is None:
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
                                messages.append(message)
                            continue
                        if event.get("type") == "system" and event.get("subtype") == "init":
                            init_seen = True
                            if not agent_id:
                                agent_id = event.get("agent_id")
                                logger.info(f"Captured agent_id {agent_id} from stream init event")
                                if progress_callback and agent_id:
                                    await progress_callback.agent_created(
                                        sample.id, agent_id=agent_id, model_handle=self.model_handle
                                    )
                            if progress_callback and agent_id:
                                await progress_callback.message_sending(
                                    sample.id, 1, len(inputs), agent_id=agent_id, model_handle=self.model_handle
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

                stderr_text = "".join(stderr_chunks)

                if process.returncode != 0:
                    # Surface the last stdout line so the real error isn't lost
                    parts = [f"Letta command failed with return code {process.returncode}"]
                    if last_line:
                        parts.append(f"Last stdout line: {last_line[:500]}")
                    if stderr_text:
                        parts.append(f"Stderr: {stderr_text[:500]}")
                    raise RuntimeError(". ".join(parts))

                if not agent_id:
                    raise RuntimeError("No agent_id found in letta stream output")

                # Extract usage from the final stream result event (best-effort).
                usage_stats = extract_usage_stats(last_line)

                if messages is not None:
                    return TargetTrace(
                        agent_id=agent_id,
                        model_handle=self.model_handle,
                        agent_usage=usage_stats,
                        context=context,
                        trajectory=[messages],
                    )
                return TargetResult(
                    agent_id=agent_id,
                    model_handle=self.model_handle,
                    agent_usage=usage_stats,
                    context=context,
                )

            except Exception as e:
                last_error = e
                attempt += 1

                if attempt > self.max_retries:
                    timeout_hint = f"Timed out after {self.timeout}s" if isinstance(e, TimeoutError) else ""
                    msg = str(e) or timeout_hint or type(e).__name__
                    err_agent_id = agent_id or factory_agent_id
                    # Best-effort: surface usage from the stream so far. Any
                    # server-side trace fields (partial trajectory/token data) are
                    # fetched by Runner from the agent_id when available.
                    partial_usage = extract_usage_stats(last_line)
                    raise TargetError(
                        msg,
                        agent_id=err_agent_id,
                        agent_usage=partial_usage,
                        trajectory=[messages] if messages else None,
                    ) from e

                backoff_time = 2 ** (attempt - 1)
                logger.warning(
                    f"Letta command failed for sample {sample.id} (attempt {attempt}/{self.max_retries + 1}). "
                    f"Agent: {agent_id or factory_agent_id or 'unknown'}. "
                    f"Error: {type(e).__name__}: {str(e)}. Retrying in {backoff_time}s..."
                )
                await anyio.sleep(backoff_time)

        raise last_error or RuntimeError("Unexpected failure in letta command retry loop")
