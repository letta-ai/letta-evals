import inspect
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import anyio
import yaml
from letta_client import AsyncLetta
from rich.console import Console

from letta_evals.datasets.loader import load_dataset
from letta_evals.graders.base import Grader
from letta_evals.graders.rubric import RubricGrader
from letta_evals.graders.tool import ToolGrader
from letta_evals.metrics import summarize_model, summarize_runs
from letta_evals.models import (
    AgentState,
    Error,
    LettaMessageUnion,
    LogicalGateSpec,
    ModelJudgeGraderSpec,
    ModelRun,
    ModelSummary,
    RunnerResult,
    Sample,
    SampleId,
    SampleResult,
    SimpleCondition,
    SimpleGateSpec,
    SuiteSpec,
    Summary,
    Timing,
    ToolGraderSpec,
    Usage,
    WeightedAverageGateSpec,
    _compare,
    normalize_weights,
)
from letta_evals.pricing import calculate_cost_from_agent_usage
from letta_evals.runner_support.grading import detect_errors, grade_sample, validate_rubric_vars
from letta_evals.sandbox.dispatch import run_sample_in_sandbox
from letta_evals.streaming import StreamingReader, StreamingWriter
from letta_evals.targets.errors import TargetError
from letta_evals.targets.letta_code_target import LettaCodeTarget
from letta_evals.types import Aggregation, ErrorCategory, LogicalOp
from letta_evals.utils import (
    extract_token_counts,
    load_object,
)
from letta_evals.visualization.base import ProgressCallback
from letta_evals.visualization.factory import ProgressStyle, create_progress_callback

logger = logging.getLogger(__name__)

# Sentinel model identifier used when a suite has no model_handles.
DEFAULT_MODEL_ID = "default"


def _model_id_for(model_handle: Optional[str]) -> str:
    """Stable bucket/path identifier for a model handle."""
    return model_handle or DEFAULT_MODEL_ID


def _build_usage(
    *,
    cost: Optional[float],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    cached_input_tokens: Optional[int],
    cache_write_tokens: Optional[int],
    reasoning_tokens: Optional[int],
) -> Optional[Usage]:
    """Build a Usage object only when there's something meaningful to record."""
    nonzero = any(v is not None and v > 0 for v in (prompt_tokens, completion_tokens, cost))
    if not nonzero:
        return None
    return Usage(
        prompt_tokens=prompt_tokens or 0,
        completion_tokens=completion_tokens or 0,
        cached_input_tokens=cached_input_tokens or 0,
        cache_write_tokens=cache_write_tokens or 0,
        reasoning_tokens=reasoning_tokens or 0,
        cost=cost if cost and cost > 0 else None,
    )


class Runner:
    """Main evaluation runner."""

    def __init__(
        self,
        suite: SuiteSpec,
        max_concurrent: int,
        progress_callback: Optional[ProgressCallback] = None,
        cached_results: Optional[RunnerResult] = None,
        output_path: Optional[Path] = None,
        letta_api_key: Optional[str] = None,
        letta_base_url: Optional[str] = None,
        letta_project_id: Optional[str] = None,
        stream_writer: Optional[StreamingWriter] = None,
        current_run: int = 1,
    ):
        self.suite: SuiteSpec = suite

        env_api_key = os.getenv("LETTA_API_KEY")
        env_base_url = os.getenv("LETTA_BASE_URL")
        env_project_id = os.getenv("LETTA_PROJECT_ID")

        api_key = letta_api_key or self.suite.target.api_key or env_api_key
        base_url = letta_base_url or self.suite.target.base_url or env_base_url
        self.project_id = letta_project_id or self.suite.target.project_id or env_project_id

        client_kwargs: dict[str, object] = {"timeout": self.suite.target.timeout}
        if base_url:
            client_kwargs["base_url"] = base_url
        elif api_key:
            client_kwargs["base_url"] = "https://api.letta.com"
            logger.info("Using default Letta Cloud base_url: https://api.letta.com")
        if api_key:
            client_kwargs["api_key"] = api_key

        logger.info(
            f"Creating AsyncLetta client with base_url={client_kwargs.get('base_url')}, has_api_key={bool(api_key)}"
        )
        self.client = AsyncLetta(**client_kwargs)

        self.graders: Optional[Dict[str, Grader]] = None
        self._init_graders()

        # results bucketed by model_id; flat list for cross-cutting computations.
        self.results_by_model: Dict[str, List[SampleResult]] = defaultdict(list)
        self.results: List[SampleResult] = []
        self.max_concurrent = max_concurrent
        self.semaphore = anyio.Semaphore(max_concurrent)
        self.progress_callback = progress_callback
        self.model_handles = self._load_model_handles()
        self.cached_results = cached_results
        self._cached_trajectories: Dict[SampleId, Dict[str, SampleResult]] = (
            self._build_trajectory_cache() if cached_results else {}
        )
        self._sample_lookup: Dict[SampleId, Sample] = {}
        # ``stream_writer`` is shared across runs in multi-run mode; if not
        # provided, the runner builds its own. ``current_run`` is the 1-based
        # index of this run inside that shared writer.
        self.stream_writer: Optional[StreamingWriter] = stream_writer
        self.current_run: int = current_run
        self.output_path = output_path

    # ── partition helpers ──

    @property
    def model_ids(self) -> List[str]:
        return [_model_id_for(handle) for handle in self.model_handles]

    @property
    def grader_keys(self) -> List[str]:
        return list(self.graders.keys()) if self.graders else []

    # ── target setup ──

    def _load_model_handles(self) -> List[Optional[str]]:
        """Load model handles if specified."""
        if self.suite.target.model_handles is None:
            return [None]
        return list(self.suite.target.model_handles)

    def _create_letta_code_target(self, model_handle: Optional[str] = None) -> LettaCodeTarget:
        """Create the letta_code target for a model handle."""
        if not model_handle:
            raise ValueError("LettaCodeTarget requires a model_handle (string), but got None")

        return LettaCodeTarget(
            client=self.client,
            model_handle=model_handle,
            allowed_tools=self.suite.target.allowed_tools,
            disallowed_tools=self.suite.target.disallowed_tools,
            timeout=int(self.suite.target.timeout),
            max_retries=self.suite.target.max_retries,
            base_url=self.suite.target.base_url,
            agent_script=self.suite.target.agent_script,
            base_dir=self.suite.target.base_dir,
            flags=self.suite.target.flags,
            permission_mode=self.suite.target.permission_mode,
        )

    def _init_graders(self) -> None:
        """Initialize grader(s) from spec."""
        if self.suite.graders:
            self.graders = {}
            for key, gspec in self.suite.graders.items():
                if isinstance(gspec, ToolGraderSpec):
                    self.graders[key] = ToolGrader(
                        function=gspec.function,
                        extractor=gspec.extractor,
                        extractor_config=gspec.extractor_config,
                        base_dir=gspec.base_dir,
                    )
                elif isinstance(gspec, ModelJudgeGraderSpec):
                    self.graders[key] = RubricGrader(
                        prompt=gspec.prompt,
                        model=gspec.model,
                        temperature=gspec.temperature,
                        provider=gspec.provider,
                        max_retries=gspec.max_retries,
                        timeout=gspec.timeout,
                        extractor=gspec.extractor,
                        extractor_config=gspec.extractor_config,
                        base_dir=gspec.base_dir,
                        system_prompt=gspec.system_prompt,
                    )
                else:
                    raise ValueError(f"Unknown grader spec type: {type(gspec)}")
        else:
            raise ValueError("Suite must define 'graders'")

    def _requires_agent_state(self) -> bool:
        if self.graders:
            return any(grader.requires_agent_state for grader in self.graders.values())
        return False

    def _should_cleanup_agent(self) -> bool:
        return self.suite.cleanup

    # ── setup ──

    async def _run_setup(self, model_handle: Optional[str] = None) -> None:
        if not self.suite.setup_script:
            return

        try:
            setup_func = load_object(self.suite.setup_script, self.suite.base_dir)
            if not hasattr(setup_func, "_is_suite_setup"):
                raise ValueError(f"Setup function must be decorated with @suite_setup: {self.suite.setup_script}")

            param_count = getattr(setup_func, "_suite_setup_param_count", 1)

            log_msg = f"Running setup script: {self.suite.setup_script}"
            if model_handle and param_count == 2:
                log_msg += f" for model: {model_handle}"
            logger.info(log_msg)

            if inspect.iscoroutinefunction(setup_func):
                if param_count == 2:
                    await setup_func(self.client, model_handle)
                elif param_count == 1:
                    await setup_func(self.client)
                else:
                    await setup_func()
            else:
                if param_count == 2:
                    setup_func(self.client, model_handle)
                elif param_count == 1:
                    setup_func(self.client)
                else:
                    setup_func()

            logger.info("Setup completed successfully")

        except Exception as e:
            logger.error(f"Error running setup script: {e}")
            raise RuntimeError(f"Setup failed: {e}") from e

    # ── cache ──

    def _build_trajectory_cache(self) -> Dict[SampleId, Dict[str, SampleResult]]:
        """Cache previous results indexed by sample_id → model → SampleResult."""
        cache: Dict[SampleId, Dict[str, SampleResult]] = defaultdict(dict)
        if self.cached_results:
            for model_id, model_run in self.cached_results.runs.items():
                for result in model_run.results:
                    cache[result.sample_id][model_id] = result
        return cache

    async def _get_or_run_trajectory(
        self,
        sample: Sample,
        model_handle: Optional[str],
        retrieve_agent_state: bool = False,
        return_token_data: bool = False,
    ) -> tuple[
        List[List[LettaMessageUnion]],
        str,
        str,
        Optional[list[dict]],
        Optional[AgentState],
        Optional[list],
    ]:
        """Return (trajectory, agent_id, model_handle, agent_usage, agent_state, token_data)."""
        sample_id = sample.id

        if self.cached_results:
            cached_result: Optional[SampleResult] = None
            cached_models = self._cached_trajectories.get(sample_id)

            if cached_models:
                lookup_key = model_handle or DEFAULT_MODEL_ID
                if lookup_key in cached_models:
                    cached_result = cached_models[lookup_key]
                elif len(cached_models) == 1:
                    cached_result = next(iter(cached_models.values()))

            if cached_result is not None:
                if self.progress_callback:
                    await self.progress_callback.agent_created(
                        sample_id, agent_id=cached_result.agent_id, model_handle=model_handle, from_cache=True
                    )
                return (
                    cached_result.trajectory,
                    cached_result.agent_id,
                    model_handle or DEFAULT_MODEL_ID,
                    getattr(cached_result, "agent_usage", None),
                    getattr(cached_result, "agent_state", None),
                    getattr(cached_result, "token_data", None),
                )

        target = self._create_letta_code_target(model_handle)
        target_result = await target.run(
            sample,
            progress_callback=self.progress_callback,
            project_id=self.project_id,
            retrieve_agent_state=retrieve_agent_state,
            return_token_data=return_token_data,
        )
        return (
            target_result.trajectory,
            target_result.agent_id,
            target_result.model_handle,
            target_result.agent_usage,
            target_result.agent_state,
            target_result.token_data,
        )

    # ── per-sample driver ──

    async def run_sample(
        self,
        sample: Sample,
        model_handle: Optional[str] = None,
        return_token_data: bool = False,
    ) -> SampleResult:
        """Run a single sample through target and grader."""
        sample_id = sample.id

        async with self.semaphore:
            agent_id = None
            agent_usage = None
            phase = ErrorCategory.UNKNOWN
            t_sample_start = time.perf_counter()
            try:  # noqa: SIM105 — outer try/finally for agent cleanup
                if self.progress_callback:
                    await self.progress_callback.sample_started(sample_id, model_handle=model_handle)

                if self.suite.sandbox is not None:
                    result = await run_sample_in_sandbox(
                        self.suite, sample, model_handle, return_token_data, t_sample_start
                    )
                    # Fire post-completion callbacks based on the final result —
                    # mid-sample events (grading_started, token streaming) are
                    # not emitted in v1 because the host only sees the final
                    # SampleResult JSON.
                    if self.progress_callback:
                        cost = result.usage.cost if (result.usage and result.usage.cost) else None
                        if result.error is not None:
                            await self.progress_callback.sample_error(
                                sample_id,
                                result.error.message,
                                agent_id=result.agent_id,
                                model_handle=model_handle,
                                target_cost=cost,
                            )
                        else:
                            primary_score = next(iter(result.grades.values())).score if result.grades else 0.0
                            primary_rationale = next(iter(result.grades.values())).rationale if result.grades else None
                            metric_scores = {k: v.score for k, v in result.grades.items()}
                            metric_rationales = {k: (v.rationale or "") for k, v in result.grades.items()}
                            await self.progress_callback.sample_completed(
                                sample_id,
                                agent_id=result.agent_id,
                                score=primary_score,
                                target_cost=cost,
                                model_handle=model_handle,
                                metric_scores=metric_scores,
                                rationale=primary_rationale,
                                metric_rationales=metric_rationales,
                            )
                    return result

                phase = ErrorCategory.TARGET
                retrieve_agent_state = self._requires_agent_state()
                (
                    trajectory,
                    agent_id,
                    model_handle,
                    agent_usage,
                    agent_state,
                    token_data,
                ) = await self._get_or_run_trajectory(
                    sample,
                    model_handle,
                    retrieve_agent_state=retrieve_agent_state,
                    return_token_data=return_token_data,
                )

                cost = calculate_cost_from_agent_usage(model_handle, agent_usage) if model_handle else None
                prompt_tokens, completion_tokens, cached_input_tokens, cache_write_tokens, reasoning_tokens = (
                    extract_token_counts(agent_usage)
                )
                target_time = time.perf_counter() - t_sample_start

                if self.progress_callback:
                    await self.progress_callback.grading_started(
                        sample_id, agent_id=agent_id, model_handle=model_handle
                    )

                phase = ErrorCategory.GRADING
                grades_dict, submissions_dict, per_grader_time = await grade_sample(
                    sample,
                    trajectory,
                    agent_state,
                    self.graders,
                    sample_id,
                    agent_id,
                    model_handle,
                    self.progress_callback,
                )

                error = detect_errors(grades_dict, trajectory, submissions_dict)
                primary_score = next(iter(grades_dict.values())).score if grades_dict else 0.0
                primary_rationale = next(iter(grades_dict.values())).rationale if grades_dict else None

                if error and self.progress_callback:
                    await self.progress_callback.sample_error(
                        sample_id,
                        error.message,
                        agent_id=agent_id,
                        model_handle=model_handle,
                        target_cost=cost if cost and cost > 0 else None,
                    )

                if error is None and self.progress_callback:
                    metric_scores = {k: v.score for k, v in grades_dict.items()}
                    metric_rationales = {k: (v.rationale or "") for k, v in grades_dict.items()}
                    await self.progress_callback.sample_completed(
                        sample_id,
                        agent_id=agent_id,
                        score=primary_score,
                        target_cost=cost if cost and cost > 0 else None,
                        model_handle=model_handle,
                        metric_scores=metric_scores,
                        rationale=primary_rationale,
                        metric_rationales=metric_rationales,
                    )

                total_time = time.perf_counter() - t_sample_start
                extraction_time = sum(gr.metadata.get("extraction_time", 0.0) for gr in grades_dict.values())

                usage = _build_usage(
                    cost=cost,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_input_tokens=cached_input_tokens,
                    cache_write_tokens=cache_write_tokens,
                    reasoning_tokens=reasoning_tokens,
                )
                timing = Timing(
                    total=total_time,
                    target=target_time,
                    extraction=extraction_time if extraction_time > 0 else None,
                    per_grader=per_grader_time if per_grader_time else None,
                )

                return SampleResult(
                    sample_id=sample_id,
                    agent_id=agent_id,
                    trajectory=trajectory,
                    submissions=submissions_dict,
                    grades=grades_dict,
                    usage=usage,
                    timing=timing,
                    error=error,
                    agent_usage=agent_usage,
                    agent_state=agent_state,
                    token_data=token_data,
                )
            except Exception as e:
                if isinstance(e, TargetError) and e.agent_id:
                    agent_id = e.agent_id
                agent_str = f" ({agent_id})" if agent_id else ""
                log_message = str(e) or type(e).__name__
                logger.error(f"Error running sample {sample_id}{agent_str} with model {model_handle}: {log_message}")
                category = ErrorCategory.TARGET if isinstance(e, TargetError) else phase
                cause = e.__cause__ if e.__cause__ else e
                error_message = str(e) or type(cause).__name__
                error = Error(
                    category=category,
                    exception_type=type(cause).__name__,
                    message=error_message,
                )
                result_model_handle = locals().get("model_handle") or model_handle
                cost = (
                    calculate_cost_from_agent_usage(result_model_handle, agent_usage)
                    if result_model_handle and agent_usage
                    else None
                )
                prompt_tokens, completion_tokens, cached_input_tokens, cache_write_tokens, reasoning_tokens = (
                    extract_token_counts(agent_usage)
                )
                if self.progress_callback:
                    await self.progress_callback.sample_error(
                        sample_id,
                        error_message,
                        agent_id=agent_id,
                        model_handle=result_model_handle,
                        target_cost=cost if cost and cost > 0 else None,
                    )
                usage = _build_usage(
                    cost=cost,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_input_tokens=cached_input_tokens,
                    cache_write_tokens=cache_write_tokens,
                    reasoning_tokens=reasoning_tokens,
                )
                timing = Timing(
                    total=time.perf_counter() - t_sample_start,
                    target=0.0,
                )
                return SampleResult(
                    sample_id=sample_id,
                    agent_id=agent_id,
                    trajectory=[],
                    submissions={},
                    grades={},
                    usage=usage,
                    timing=timing,
                    error=error,
                    agent_usage=agent_usage,
                )
            finally:
                if self._should_cleanup_agent() and agent_id:
                    try:
                        await self.client.agents.delete(agent_id=agent_id)
                        logger.info(f"Cleaned up agent {agent_id} for sample {sample_id}")
                    except Exception as cleanup_err:
                        logger.warning(f"Failed to cleanup agent {agent_id}: {cleanup_err}")

    # ── orchestration ──

    async def run(self) -> RunnerResult:
        """Run evaluation across all configured models and samples (single run).

        For multi-run mode the orchestrator (``_execute_runs``) constructs a
        Runner per run and shares one StreamingWriter across them.
        """
        # Setup-once (when setup doesn't need model_handle).
        setup_needs_model = False
        if self.suite.setup_script:
            setup_func = load_object(self.suite.setup_script, self.suite.base_dir)
            param_count = getattr(setup_func, "_suite_setup_param_count", 1)
            setup_needs_model = param_count == 2

        if not setup_needs_model:
            await self._run_setup()

        samples = list(
            load_dataset(self.suite.dataset, max_samples=self.suite.max_samples, sample_tags=self.suite.sample_tags)
        )
        self._sample_lookup = {s.id: s for s in samples}

        validate_rubric_vars(self.suite, samples)

        self.results_by_model.clear()
        self.results = []

        # Build the writer if not provided externally.
        owns_writer = False
        if self.stream_writer is None and self.output_path:
            self.stream_writer = StreamingWriter(
                self.output_path,
                suite_spec=self.suite,
                samples=samples,
                models=self.model_ids,
                num_runs=1,
            )
            await self.stream_writer.initialize()
            owns_writer = True

        try:
            async with anyio.create_task_group() as tg:
                for model_handle in self.model_handles:
                    if setup_needs_model:
                        await self._run_setup(model_handle=model_handle)

                    model_id = _model_id_for(model_handle)

                    for sample in samples:

                        async def run_and_append(s, cfg, mid):
                            result = await self.run_sample(s, model_handle=cfg)
                            self.results_by_model[mid].append(result)
                            self.results.append(result)
                            if self.stream_writer:
                                await self.stream_writer.append_result(result, model=mid, run=self.current_run)

                        tg.start_soon(run_and_append, sample, model_handle, model_id)

            # Per-model summaries
            model_summaries: List[ModelSummary] = []
            for model_id in self.model_ids:
                ms = summarize_model(
                    model=model_id,
                    results=self.results_by_model.get(model_id, []),
                    grader_keys=self.grader_keys,
                    gate=self.suite.gate,
                )
                model_summaries.append(ms)

            gates_passed = self._check_gates_flat(self.results)
            summary = Summary(
                suite=self.suite.name,
                models=model_summaries,
                gates_passed=gates_passed,
            )

            # Build per-model in-memory runs.
            runs: Dict[str, ModelRun] = {}
            for ms in model_summaries:
                runs[ms.model] = ModelRun(
                    model=ms.model,
                    results=self.results_by_model.get(ms.model, []),
                    summary=ms,
                )

            if self.stream_writer and owns_writer:
                await self.stream_writer.write_summary(summary)

            return RunnerResult(
                suite_spec=self.suite,
                samples=samples,
                runs=runs,
                summary=summary,
                gates_passed=gates_passed,
            )
        except BaseException:
            # Best-effort summary on interruption.
            try:
                if self.stream_writer and owns_writer:
                    model_summaries = []
                    for model_id in self.model_ids:
                        model_summaries.append(
                            summarize_model(
                                model=model_id,
                                results=self.results_by_model.get(model_id, []),
                                grader_keys=self.grader_keys,
                                gate=self.suite.gate,
                            )
                        )
                    gates_passed = self._check_gates_flat(self.results)
                    summary = Summary(
                        suite=self.suite.name,
                        models=model_summaries,
                        gates_passed=gates_passed,
                    )
                    await self.stream_writer.write_summary(summary)
            finally:
                raise

    # ── gate evaluation ──

    def _compute_aggregation(
        self,
        results: List[SampleResult],
        metric_key: str,
        aggregation: Aggregation,
        pass_threshold: Optional[float] = None,
    ) -> float:
        scores = [r.grades[metric_key].score for r in results if metric_key in r.grades]

        if not scores:
            return 0.0

        if aggregation == Aggregation.AVG_SCORE:
            return sum(scores) / len(scores)
        elif aggregation == Aggregation.MIN:
            return min(scores)
        elif aggregation == Aggregation.MAX:
            return max(scores)
        elif aggregation in (Aggregation.MEDIAN, Aggregation.P50):
            import statistics

            return statistics.median(scores)
        elif aggregation in (Aggregation.P95, Aggregation.P99):
            import numpy as np

            percentile = 95 if aggregation == Aggregation.P95 else 99
            return float(np.percentile(scores, percentile))
        elif aggregation == Aggregation.ACCURACY:
            threshold = pass_threshold if pass_threshold is not None else 1.0
            passed = sum(1 for s in scores if s >= threshold)
            return (passed / len(scores)) * 100.0
        else:
            return 0.0

    def _evaluate_simple_condition(self, results: List[SampleResult], condition: SimpleCondition) -> bool:
        if condition.metric_key not in self.graders:
            raise ValueError(f"metric_key '{condition.metric_key}' not found in graders")
        value = self._compute_aggregation(
            results, condition.metric_key, condition.aggregation, condition.pass_threshold
        )
        return _compare(value, condition.op, condition.value)

    def _evaluate_logical_gate(self, results: List[SampleResult], gate: LogicalGateSpec) -> bool:
        sub_results = []
        for condition in gate.conditions:
            if isinstance(condition, SimpleCondition):
                sub_results.append(self._evaluate_simple_condition(results, condition))
            elif isinstance(condition, LogicalGateSpec):
                sub_results.append(self._evaluate_logical_gate(results, condition))
            else:
                raise ValueError(f"unknown condition type: {type(condition)}")

        if gate.operator == LogicalOp.AND:
            return all(sub_results)
        elif gate.operator == LogicalOp.OR:
            return any(sub_results)
        else:
            raise ValueError(f"unknown logical operator: {gate.operator}")

    def _check_gates_flat(self, results: List[SampleResult]) -> bool:
        """Evaluate the suite's gate against a flat list of sample results."""
        gate = self.suite.gate

        if isinstance(gate, SimpleGateSpec):
            if gate.metric_key not in self.graders:
                raise ValueError(f"metric_key '{gate.metric_key}' not found in graders")
            value = self._compute_aggregation(results, gate.metric_key, gate.aggregation, gate.pass_threshold)
            return _compare(value, gate.op, gate.value)

        elif isinstance(gate, WeightedAverageGateSpec):
            for metric_key in gate.weights.keys():
                if metric_key not in self.graders:
                    raise ValueError(f"metric_key '{metric_key}' not found in graders")

            normalized = normalize_weights(gate.weights)
            weighted_sum = 0.0
            for metric_key, weight in normalized.items():
                agg_value = self._compute_aggregation(results, metric_key, gate.aggregation)
                weighted_sum += weight * agg_value

            return _compare(weighted_sum, gate.op, gate.value)

        elif isinstance(gate, LogicalGateSpec):
            return self._evaluate_logical_gate(results, gate)

        else:
            raise ValueError(f"unknown gate type: {type(gate)}")


# ── top-level run_suite ──


async def run_suite(
    suite_path: Path,
    max_concurrent: int,
    *,
    custom_progress_callback: Optional[ProgressCallback] = None,
    progress_style: ProgressStyle | str = ProgressStyle.NONE,
    cached_results_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    letta_api_key: Optional[str] = None,
    letta_base_url: Optional[str] = None,
    letta_project_id: Optional[str] = None,
    num_runs: Optional[int] = None,
) -> RunnerResult:
    """Load and run a suite from YAML file."""
    if custom_progress_callback is not None:
        style_val = progress_style if isinstance(progress_style, ProgressStyle) else ProgressStyle(progress_style)
        if style_val != ProgressStyle.NONE:
            raise ValueError(
                "Cannot specify both 'custom_progress_callback' and 'progress_style'. "
                "Use custom_progress_callback for custom implementations, or progress_style for built-in styles."
            )

    with open(suite_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    suite = SuiteSpec.from_yaml(yaml_data, base_dir=suite_path.parent)

    actual_num_runs = num_runs if num_runs is not None else (suite.num_runs or 1)

    if actual_num_runs > 1 and cached_results_path:
        raise ValueError("Cannot use --num-runs > 1 with --cached (results would be identical)")

    cached_results = None
    if cached_results_path:
        if not cached_results_path.exists():
            raise ValueError(f"Cached results file not found: {cached_results_path}")

        cached_results = await StreamingReader.to_runner_result(cached_results_path)

        cached_sample_map = {s.id: s for s in cached_results.samples}
        samples = list(load_dataset(suite.dataset, max_samples=suite.max_samples, sample_tags=suite.sample_tags))

        for sample in samples:
            if sample.id in cached_sample_map:
                cached_sample = cached_sample_map[sample.id]
                if cached_sample.input != sample.input:
                    raise ValueError(
                        f"Sample ID {sample.id} input mismatch: dataset has '{sample.input}' but cache has '{cached_sample.input}'"
                    )

    samples = list(load_dataset(suite.dataset, max_samples=suite.max_samples, sample_tags=suite.sample_tags))
    if suite.target.model_handles:
        num_models = len(suite.target.model_handles)
    else:
        num_models = 1
    total_evaluations = len(samples) * num_models

    metric_labels = None
    if suite.graders:
        metric_labels = {key: (gspec.display_name or key) for key, gspec in suite.graders.items()}

    if custom_progress_callback is not None:
        progress_cb = custom_progress_callback
    else:
        style_val = progress_style
        if isinstance(style_val, str):
            try:
                style_val = ProgressStyle(style_val)
            except ValueError:
                style_val = ProgressStyle.NONE
        progress_cb = create_progress_callback(
            style=style_val,  # type: ignore[arg-type]
            suite=suite,
            total_evaluations=total_evaluations,
            console=Console() if style_val == ProgressStyle.RICH else None,
            max_concurrent=max_concurrent,
            cached_mode=(cached_results_path is not None),
            metric_labels=metric_labels,
        )

    file_handler: Optional[logging.FileHandler] = None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        log_path = output_path / "run.log"
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        pkg_logger = logging.getLogger("letta_evals")
        pkg_logger.addHandler(file_handler)
        if pkg_logger.level == logging.NOTSET:
            pkg_logger.setLevel(logging.WARNING)

    try:
        return await _execute_runs(
            suite=suite,
            samples=samples,
            actual_num_runs=actual_num_runs,
            max_concurrent=max_concurrent,
            progress_cb=progress_cb,
            cached_results=cached_results,
            output_path=output_path,
            letta_api_key=letta_api_key,
            letta_base_url=letta_base_url,
            letta_project_id=letta_project_id,
        )
    finally:
        if file_handler is not None:
            logging.getLogger("letta_evals").removeHandler(file_handler)
            file_handler.close()


async def _execute_runs(
    suite: SuiteSpec,
    samples: List[Sample],
    actual_num_runs: int,
    max_concurrent: int,
    progress_cb: Optional[ProgressCallback],
    cached_results: Optional[RunnerResult],
    output_path: Optional[Path],
    letta_api_key: Optional[str],
    letta_base_url: Optional[str],
    letta_project_id: Optional[str],
) -> RunnerResult:
    """Execute single or multiple evaluation runs sharing one streaming writer."""
    if suite.target.model_handles:
        model_ids = list(suite.target.model_handles or [])
    else:
        model_ids = [DEFAULT_MODEL_ID]

    # Build a single writer that lives across all runs.
    shared_writer: Optional[StreamingWriter] = None
    if output_path:
        shared_writer = StreamingWriter(
            output_path,
            suite_spec=suite,
            samples=samples,
            models=model_ids,
            num_runs=actual_num_runs,
        )
        await shared_writer.initialize()

    # Accumulate per-run results per model for multi-run aggregation.
    per_run_results: Dict[str, List[List[SampleResult]]] = defaultdict(list)
    runs_passed = 0
    last_runner_result: Optional[RunnerResult] = None

    for run_idx in range(actual_num_runs):
        runner = Runner(
            suite,
            max_concurrent=max_concurrent,
            progress_callback=progress_cb,
            cached_results=cached_results,
            output_path=output_path,
            letta_api_key=letta_api_key,
            letta_base_url=letta_base_url,
            letta_project_id=letta_project_id,
            stream_writer=shared_writer,
            current_run=run_idx + 1,
        )

        if progress_cb is not None:
            if run_idx == 0:
                await progress_cb.start()
            else:
                progress_cb.reset()

        try:
            result = await runner.run()
            last_runner_result = result
            for model_id, model_run in result.runs.items():
                per_run_results[model_id].append(model_run.results)
            if result.gates_passed:
                runs_passed += 1
        finally:
            if progress_cb is not None and run_idx == actual_num_runs - 1:
                progress_cb.stop()

    if last_runner_result is None:
        raise RuntimeError("No runs completed")

    # Build the final aggregated summary.
    grader_keys = list(suite.graders.keys()) if suite.graders else []

    if actual_num_runs > 1:
        aggregated_models: List[ModelSummary] = []
        for model_id in model_ids:
            ms = summarize_runs(
                model=model_id,
                per_run_results=per_run_results.get(model_id, []),
                grader_keys=grader_keys,
                gate=suite.gate,
            )
            aggregated_models.append(ms)
            # Write per-model summary.json with the per-run breakdown.
            if shared_writer is not None:
                await shared_writer.write_model_summary(ms)

        final_summary = Summary(
            suite=suite.name,
            models=aggregated_models,
            gates_passed=runs_passed > 0,
            runs_passed=runs_passed,
        )
        if shared_writer is not None:
            await shared_writer.write_summary(final_summary)

        # Build the in-memory RunnerResult with multi-run breakdown per model.
        final_runs: Dict[str, ModelRun] = {}
        for ms in aggregated_models:
            runs_list = per_run_results.get(ms.model, [])
            final_runs[ms.model] = ModelRun(
                model=ms.model,
                results=runs_list[-1] if runs_list else [],
                runs=runs_list if runs_list else None,
                summary=ms,
            )

        final_result = RunnerResult(
            suite_spec=suite,
            samples=samples,
            runs=final_runs,
            summary=final_summary,
            gates_passed=runs_passed > 0,
        )

        if progress_cb is not None:
            await progress_cb.suite_completed(final_result)
        return final_result

    # Single-run: last_runner_result is already the right shape, but we need to
    # write the top-level summary now that the writer is shared.
    if shared_writer is not None:
        await shared_writer.write_summary(last_runner_result.summary)
    if progress_cb is not None:
        await progress_cb.suite_completed(last_runner_result)
    return last_runner_result
