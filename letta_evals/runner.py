import inspect
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import anyio
import yaml
from letta_client import AsyncLetta, LettaMessageUnion, LlmConfig

from letta_evals.datasets.loader import load_jsonl
from letta_evals.graders.base import Grader
from letta_evals.graders.rubric import RubricGrader
from letta_evals.graders.tool import ToolGrader
from letta_evals.models import (
    GradeResult,
    MetricAggregate,
    Metrics,
    ModelMetrics,
    RunnerResult,
    Sample,
    SampleResult,
    SuiteSpec,
)
from letta_evals.streaming import StreamingReader, StreamingWriter
from letta_evals.targets.agent import AgentTarget
from letta_evals.targets.base import Target
from letta_evals.types import GateMetric, GraderKind, ProgressCallback, TargetKind
from letta_evals.utils import load_object

logger = logging.getLogger(__name__)


class Runner:
    """Main evaluation runner."""

    def __init__(
        self,
        suite: SuiteSpec,
        max_concurrent: int,
        progress_callback: Optional[ProgressCallback] = None,
        cached_results: Optional[RunnerResult] = None,
        output_path: Optional[Path] = None,
    ):
        self.suite: SuiteSpec = suite
        # Support single or multiple graders
        self.grader: Optional[Grader] = None
        self.graders: Optional[Dict[str, Grader]] = None
        self._init_graders()
        self.results: List[SampleResult] = []
        self.max_concurrent = max_concurrent
        self.semaphore = anyio.Semaphore(max_concurrent)
        self.progress_callback = progress_callback
        self.model_configs = self._load_model_configs()
        self.cached_results = cached_results
        self._cached_trajectories: Dict[int, Dict[str, SampleResult]] = (
            self._build_trajectory_cache() if cached_results else {}
        )
        self._setup_executed = False
        self.stream_writer: Optional[StreamingWriter] = None
        self.output_path = output_path

        self.client = AsyncLetta(
            base_url=self.suite.target.base_url, token=self.suite.target.api_key, timeout=self.suite.target.timeout
        )

    def _load_model_configs(self) -> List[Optional[LlmConfig]]:
        """Load model configurations if specified."""
        if not self.suite.target.model_configs:
            return [None]  # no model configs, use default

        configs = []
        model_configs_dir = Path(__file__).parent / "llm_model_configs"

        for config_name in self.suite.target.model_configs:
            config_path = model_configs_dir / f"{config_name}.json"
            if not config_path.exists():
                raise ValueError(f"Model config not found at path: {config_path}")

            with open(config_path, "r") as f:
                config_data = json.load(f)
                llm_config = LlmConfig(**config_data)
                configs.append(llm_config)

        return configs

    def _create_target(self, llm_config: Optional[LlmConfig] = None) -> Target:
        """Create target from spec, optionally with model config."""
        if self.suite.target.kind == TargetKind.AGENT:
            return AgentTarget(
                client=self.client,
                agent_id=self.suite.target.agent_id,
                agent_file=self.suite.target.agent_file,
                agent_script=self.suite.target.agent_script,
                base_dir=self.suite.target.base_dir,
                llm_config=llm_config,
            )
        else:
            raise ValueError(f"Unknown target kind: {self.suite.target.kind}")

    def _init_graders(self) -> None:
        """Initialize grader(s) from spec."""
        if self.suite.graders:
            self.graders = {}
            for key, gspec in self.suite.graders.items():
                if gspec.kind == GraderKind.TOOL:
                    self.graders[key] = ToolGrader(
                        function=gspec.function,
                        extractor=gspec.extractor,
                        extractor_config=gspec.extractor_config,
                        base_dir=gspec.base_dir,
                    )
                elif gspec.kind == GraderKind.RUBRIC:
                    self.graders[key] = RubricGrader(
                        prompt=gspec.prompt,
                        model=gspec.model,
                        temperature=gspec.temperature,
                        provider=gspec.provider,
                        extractor=gspec.extractor,
                        extractor_config=gspec.extractor_config,
                    )
                else:
                    raise ValueError(f"Unknown grader kind: {gspec.kind}")
        elif self.suite.grader:
            gspec = self.suite.grader
            if gspec.kind == GraderKind.TOOL:
                self.grader = ToolGrader(
                    function=gspec.function,
                    extractor=gspec.extractor,
                    extractor_config=gspec.extractor_config,
                    base_dir=gspec.base_dir,
                )
            elif gspec.kind == GraderKind.RUBRIC:
                self.grader = RubricGrader(
                    prompt=gspec.prompt,
                    model=gspec.model,
                    temperature=gspec.temperature,
                    provider=gspec.provider,
                    extractor=gspec.extractor,
                    extractor_config=gspec.extractor_config,
                )
            else:
                raise ValueError(f"Unknown grader kind: {gspec.kind}")
        else:
            raise ValueError("Suite must define either 'grader' or 'graders'")

    async def _run_setup(self) -> None:
        """Execute the setup function if specified."""
        if self._setup_executed:
            return

        if not self.suite.setup_script:
            return

        try:
            logger.info(f"Running setup script: {self.suite.setup_script}")
            setup_func = load_object(self.suite.setup_script, self.suite.base_dir)
            if not hasattr(setup_func, "_is_suite_setup"):
                raise ValueError(f"Setup function must be decorated with @suite_setup: {self.suite.setup_script}")

            if inspect.iscoroutinefunction(setup_func):
                await setup_func(self.client)
            else:
                setup_func(self.client)

            self._setup_executed = True
            logger.info("Setup completed successfully")

        except Exception as e:
            logger.error(f"Error running setup script: {e}")
            raise RuntimeError(f"Setup failed: {e}") from e

    def _build_trajectory_cache(self) -> Dict[int, Dict[str, SampleResult]]:
        """Build a cache of sample results indexed by sample_id -> model_name -> SampleResult."""
        cache: Dict[int, Dict[str, SampleResult]] = defaultdict(dict)
        if self.cached_results:
            for result in self.cached_results.results:
                # use model_name as key, or None if not specified
                model_key = result.model_name if result.model_name else None
                cache[result.sample.id][model_key] = result
        return cache

    async def _get_or_run_trajectory(
        self, sample: Sample, llm_config: Optional[LlmConfig]
    ) -> tuple[List[List[LettaMessageUnion]], str, str, Optional[list[dict]]]:
        """Return (trajectory, agent_id, model_name, agent_usage) using cache or by running the target.

        If cache is enabled and contains an exact match, use it; otherwise run the target.
        """
        sample_id = sample.id
        model_name = llm_config.model if llm_config else None

        if self.cached_results:
            cached_result: Optional[SampleResult] = None
            cached_models = self._cached_trajectories.get(sample_id)

            if cached_models:
                if model_name is not None:
                    cached_result = cached_models.get(model_name)
                else:
                    if len(cached_models) == 1:
                        cached_result = next(iter(cached_models.values()))
                        model_name = cached_result.model_name

            if cached_result is not None:
                if self.progress_callback:
                    await self.progress_callback.agent_loading(sample_id, model_name=model_name, from_cache=True)
                return cached_result.trajectory, cached_result.agent_id, model_name, getattr(
                    cached_result, "agent_usage", None
                )

        target = self._create_target(llm_config)
        target_result = await target.run(sample, progress_callback=self.progress_callback)
        return target_result.trajectory, target_result.agent_id, target_result.model_name, target_result.agent_usage

    async def run_sample(self, sample: Sample, llm_config: Optional[LlmConfig] = None) -> SampleResult:
        """Run a single sample through target and grader."""
        sample_id = sample.id
        model_name = llm_config.model if llm_config else None

        async with self.semaphore:
            try:
                if self.progress_callback:
                    await self.progress_callback.sample_started(sample_id, model_name=model_name)
                trajectory, agent_id, model_name, agent_usage = await self._get_or_run_trajectory(sample, llm_config)

                if self.progress_callback:
                    await self.progress_callback.grading_started(sample_id, model_name=model_name)

                grades_dict: Optional[Dict[str, GradeResult]] = None
                submissions_dict: Optional[Dict[str, str]] = None
                if self.graders is not None:
                    grades_dict = {}
                    submissions_dict = {}
                    for key, grader in self.graders.items():
                        gr, sub = await grader.grade(sample, trajectory)
                        grades_dict[key] = gr
                        submissions_dict[key] = sub
                    # Determine gating metric key
                    gate_key = self._gate_metric_key()
                    gate_grade = grades_dict.get(gate_key) if gate_key in grades_dict else next(iter(grades_dict.values()))
                    gate_submission = submissions_dict.get(gate_key) if gate_key in submissions_dict else next(
                        iter(submissions_dict.values())
                    )
                    grade_result, submission = gate_grade, gate_submission
                else:
                    grade_result, submission = await self.grader.grade(sample, trajectory)  # type: ignore[arg-type]

                if self.progress_callback:
                    passed = self._check_sample_pass(grade_result.score)
                    metric_scores = None
                    metric_pass = None
                    if self.graders is not None and grades_dict is not None:
                        metric_scores = {k: v.score for k, v in grades_dict.items()}
                        metric_pass = {k: self._check_sample_pass(v) for k, v in metric_scores.items()}
                    await self.progress_callback.sample_completed(
                        sample_id,
                        passed=passed,
                        score=grade_result.score,
                        model_name=model_name,
                        metric_scores=metric_scores,
                        metric_pass=metric_pass,
                    )

                return SampleResult(
                    sample=sample,
                    submission=submission,
                    submissions=submissions_dict,
                    trajectory=trajectory,
                    agent_id=agent_id,
                    grade=grade_result,
                    grades=grades_dict,
                    model_name=model_name,
                    agent_usage=agent_usage,
                )
            except Exception as e:
                if self.progress_callback:
                    await self.progress_callback.sample_error(sample_id, str(e), model_name=model_name)
                raise

    async def run(self) -> RunnerResult:
        """Run evaluation on all samples."""
        await self._run_setup()

        samples = list(
            load_jsonl(self.suite.dataset, max_samples=self.suite.max_samples, sample_tags=self.suite.sample_tags)
        )

        self.results = []
        # prepare config for both streaming and final result
        config: Dict[str, Any] = {
            "target": json.loads(self.suite.target.model_dump_json()),
            "gate": json.loads(self.suite.gate.model_dump_json()),
        }
        if self.suite.graders:
            config["graders"] = {k: json.loads(v.model_dump_json()) for k, v in self.suite.graders.items()}
        elif self.suite.grader:
            config["grader"] = json.loads(self.suite.grader.model_dump_json())

        # initialize streaming writer if output path is provided
        if self.output_path:
            self.stream_writer = StreamingWriter(self.output_path, self.suite.name, config)
            await self.stream_writer.initialize()

        try:
            async with anyio.create_task_group() as tg:
                for llm_config in self.model_configs:
                    for sample in samples:

                        async def run_and_append(s, cfg):
                            try:
                                result = await self.run_sample(s, llm_config=cfg)
                                self.results.append(result)
                                if self.stream_writer:
                                    await self.stream_writer.append_result(result)
                            except Exception as e:
                                model_name = cfg.model if cfg else None
                                logger.error(f"Error running sample {s.id} with model {model_name}: {e}")
                                if self.progress_callback:
                                    await self.progress_callback.sample_error(s.id, str(e), model_name=model_name)

                                error_result = SampleResult(
                                    sample=s,
                                    submission="",
                                    submissions=None,
                                    trajectory=[],
                                    agent_id=None,
                                    grade=GradeResult(score=0.0, rationale=f"Error: {str(e)[:200]}"),
                                    grades=None,
                                    model_name=model_name,
                                    agent_usage=None,
                                )
                                self.results.append(error_result)
                                if self.stream_writer:
                                    await self.stream_writer.append_result(error_result)

                        tg.start_soon(run_and_append, sample, llm_config)

            metrics = self._calculate_metrics()
            gates_passed = self._check_gates(metrics)

            # write final metrics if streaming
            if self.stream_writer:
                await self.stream_writer.write_metrics(metrics, gates_passed)

            return RunnerResult(
                suite=self.suite.name, config=config, results=self.results, metrics=metrics, gates_passed=gates_passed
            )
        except BaseException:
            # On interruption or errors, write a best-effort summary for a valid JSONL
            try:
                metrics = self._calculate_metrics()
                gates_passed = self._check_gates(metrics)
                if self.stream_writer:
                    await self.stream_writer.write_metrics(metrics, gates_passed)
            finally:
                # Re-raise to preserve original error/interrupt semantics
                raise

    def _calculate_metrics(self) -> Metrics:
        """Calculate aggregate metrics from results.

        - total: success + error (all results)
        - total_attempted: success only (completed without error)
        - accuracy: percent of attempted that passed the gate (based on configured per-sample pass)
        - avg_score: mean across all results (including error results)
        - per_model: same semantics per model (based on gate metric key)
        """
        total = len(self.results)
        if total == 0:
            return Metrics(total=0, total_attempted=0, avg_score=0.0, accuracy=0.0, passed_attempts=0, failed_attempts=0)

        # success = completed without error; error results have empty trajectory or missing agent_id
        def is_success(r: SampleResult) -> bool:
            return (r.agent_id is not None) and bool(r.trajectory)

        attempted = sum(1 for r in self.results if is_success(r))

        # Determine per-metric aggregates if multiple graders
        by_metric: Dict[str, MetricAggregate] = {}
        if self.graders is not None:
            for metric_key in self.graders.keys():
                m_scores = [r.grades[metric_key].score for r in self.results if r.grades and metric_key in r.grades]
                m_avg = sum(m_scores) / len(m_scores) if m_scores else 0.0
                m_passed = sum(
                    1
                    for r in self.results
                    if is_success(r)
                    and r.grades
                    and metric_key in r.grades
                    and self._check_sample_pass(r.grades[metric_key].score)
                )
                m_accuracy = (m_passed / attempted) * 100.0 if attempted > 0 else 0.0
                by_metric[metric_key] = MetricAggregate(
                    avg_score=m_avg, accuracy=m_accuracy, passed_attempts=m_passed, failed_attempts=(attempted - m_passed)
                )

        # Choose base metric values for top-level fields
        if self.graders is not None:
            gate_key = self._gate_metric_key()
            agg = by_metric.get(gate_key) if gate_key in by_metric else (next(iter(by_metric.values())) if by_metric else None)
            avg_score = agg.avg_score if agg else 0.0
            passed_attempts = agg.passed_attempts if agg else 0
            accuracy = agg.accuracy if agg else 0.0
        else:
            scores = [r.grade.score for r in self.results]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            passed_attempts = sum(1 for r in self.results if is_success(r) and self._check_sample_pass(r.grade.score))
            accuracy = (passed_attempts / attempted) * 100.0 if attempted > 0 else 0.0

        per_model = None
        if self.suite.target.model_configs:
            model_results = defaultdict(list)
            for result in self.results:
                model_results[result.model_name].append(result)

            per_model = []
            for model_name, results in model_results.items():
                model_attempted = sum(1 for r in results if is_success(r))
                if self.graders is not None:
                    gate_key = self._gate_metric_key()
                    model_scores = [
                        r.grades[gate_key].score for r in results if r.grades and gate_key in r.grades
                    ]
                    model_passed = sum(
                        1
                        for r in results
                        if is_success(r) and r.grades and gate_key in r.grades and self._check_sample_pass(r.grades[gate_key].score)
                    )
                else:
                    model_scores = [r.grade.score for r in results]
                    model_passed = sum(1 for r in results if is_success(r) and self._check_sample_pass(r.grade.score))
                model_avg = sum(model_scores) / len(model_scores) if model_scores else 0.0
                model_accuracy = (model_passed / model_attempted) * 100.0 if model_attempted > 0 else 0.0

                per_model.append(
                    ModelMetrics(
                        model_name=model_name,
                        total=len(results),
                        total_attempted=model_attempted,
                        avg_score=model_avg,
                        passed_samples=model_passed,
                        failed_samples=(model_attempted - model_passed),
                        accuracy=model_accuracy,
                    )
                )

        return Metrics(
            total=total,
            total_attempted=attempted,
            avg_score=avg_score,
            accuracy=accuracy,
            passed_attempts=passed_attempts,
            failed_attempts=(attempted - passed_attempts),
            per_model=per_model,
            by_metric=by_metric if by_metric else None,
        )

    def _check_sample_pass(self, score: float) -> bool:
        """Check if an individual score satisfies the per-sample pass criteria."""
        return self.suite.gate.check_sample(score)

    def _check_gates(self, metrics: Metrics) -> bool:
        """Check if the configured gate metric is satisfied."""
        metric_kind = self.suite.gate.metric
        # determine which metric key (grader) we're gating on
        gate_key = self._gate_metric_key()
        # derive value from either per-metric or top-level
        if self.graders is not None:
            # recompute a lightweight aggregate for gate metric from current results to avoid dependency
            if metric_kind == GateMetric.AVG_SCORE:
                scores = [
                    r.grades[gate_key].score
                    for r in self.results
                    if r.grades and gate_key in r.grades
                ]
                value = (sum(scores) / len(scores)) if scores else 0.0
            elif metric_kind == GateMetric.ACCURACY:
                # accuracy over attempted
                def is_success(r: SampleResult) -> bool:
                    return (r.agent_id is not None) and bool(r.trajectory)

                attempted = sum(1 for r in self.results if is_success(r))
                passed = sum(
                    1
                    for r in self.results
                    if is_success(r) and r.grades and gate_key in r.grades and self._check_sample_pass(r.grades[gate_key].score)
                )
                value = (passed / attempted) * 100.0 if attempted > 0 else 0.0
            else:
                value = 0.0
        else:
            value = metrics.avg_score if metric_kind == GateMetric.AVG_SCORE else metrics.accuracy
        return self.suite.gate._compare(value, self.suite.gate.op, self.suite.gate.value)

    def _gate_metric_key(self) -> str:
        """Return the selected metric key (grader name) for gating.

        If not specified, uses the only grader if single, otherwise the first in order.
        """
        if self.suite.gate.metric_key:
            return self.suite.gate.metric_key
        if self.graders is not None and len(self.graders) > 0:
            # return first key (deterministic by insertion order)
            return next(iter(self.graders.keys()))
        return "default"


async def run_suite(
    suite_path: Path,
    max_concurrent: int,
    progress_callback: Optional[ProgressCallback] = None,
    cached_results_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> RunnerResult:
    """Load and run a suite from YAML file."""
    with open(suite_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    suite = SuiteSpec.from_yaml(yaml_data, base_dir=suite_path.parent)

    cached_results = None
    if cached_results_path:
        if not cached_results_path.exists():
            raise ValueError(f"Cached results file not found: {cached_results_path}")

        # cached files are now in JSONL streaming format
        cached_results = await StreamingReader.to_runner_result(cached_results_path)

        cached_sample_map = {result.sample.id: result.sample for result in cached_results.results}
        samples = list(load_jsonl(suite.dataset, max_samples=suite.max_samples, sample_tags=suite.sample_tags))

        for sample in samples:
            if sample.id in cached_sample_map:
                cached_sample = cached_sample_map[sample.id]
                if cached_sample.input != sample.input:
                    raise ValueError(
                        f"Sample ID {sample.id} input mismatch: dataset has '{sample.input}' but cache has '{cached_sample.input}'"
                    )

    runner = Runner(
        suite,
        max_concurrent=max_concurrent,
        progress_callback=progress_callback,
        cached_results=cached_results,
        output_path=output_path,
    )
    return await runner.run()
