import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import anyio
import yaml
from letta_client import LlmConfig

from letta_evals.datasets.loader import load_jsonl
from letta_evals.graders.base import Grader
from letta_evals.graders.rubric import RubricGrader
from letta_evals.graders.tool import ToolGrader
from letta_evals.models import Metrics, ModelMetrics, RunnerResult, Sample, SampleResult, SuiteSpec
from letta_evals.targets.agent import AgentTarget
from letta_evals.targets.base import Target
from letta_evals.types import GraderKind, ProgressCallback, TargetKind


class Runner:
    """Main evaluation runner."""

    def __init__(
        self,
        suite: SuiteSpec,
        max_concurrent: int,
        progress_callback: Optional[ProgressCallback] = None,
        cached_results: Optional[RunnerResult] = None,
    ):
        self.suite: SuiteSpec = suite
        self.grader: Grader = self._create_grader()
        self.results: List[SampleResult] = []
        self.max_concurrent = max_concurrent
        self.semaphore = anyio.Semaphore(max_concurrent)
        self.progress_callback = progress_callback
        self.model_configs = self._load_model_configs()
        self.cached_results = cached_results
        self._cached_trajectories: Dict[int, Dict[str, SampleResult]] = (
            self._build_trajectory_cache() if cached_results else {}
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
                base_url=self.suite.target.base_url,
                agent_id=self.suite.target.agent_id,
                agent_file=self.suite.target.agent_file,
                agent_script=self.suite.target.agent_script,
                api_key=self.suite.target.api_key,
                timeout=self.suite.target.timeout,
                base_dir=self.suite.target.base_dir,
                llm_config=llm_config,
            )
        else:
            raise ValueError(f"Unknown target kind: {self.suite.target.kind}")

    def _create_grader(self) -> Grader:
        """Create grader from spec."""
        if self.suite.grader.kind == GraderKind.TOOL:
            return ToolGrader(
                function=self.suite.grader.function,
                extractor=self.suite.grader.extractor,
                extractor_config=self.suite.grader.extractor_config,
                base_dir=self.suite.grader.base_dir,
            )
        elif self.suite.grader.kind == GraderKind.RUBRIC:
            return RubricGrader(
                prompt=self.suite.grader.prompt,
                model=self.suite.grader.model,
                temperature=self.suite.grader.temperature,
                provider=self.suite.grader.provider,
                extractor=self.suite.grader.extractor,
                extractor_config=self.suite.grader.extractor_config,
            )
        else:
            raise ValueError(f"Unknown grader kind: {self.suite.grader.kind}")

    def _build_trajectory_cache(self) -> Dict[int, Dict[str, SampleResult]]:
        """Build a cache of sample results indexed by sample_id -> model_name -> SampleResult."""
        cache: Dict[int, Dict[str, SampleResult]] = defaultdict(dict)
        if self.cached_results:
            for result in self.cached_results.results:
                cache[result.sample.id][result.model_name] = result
        return cache

    async def run_sample(self, sample: Sample, llm_config: Optional[LlmConfig] = None) -> SampleResult:
        """Run a single sample through target and grader."""
        use_cached = self.cached_results is not None
        sample_id = sample.id

        async with self.semaphore:
            try:
                model_name = llm_config.model if llm_config else None

                if use_cached:
                    # check if we have cached results for this sample
                    if sample_id not in self._cached_trajectories:
                        raise ValueError(f"Cached trajectory not found for sample {sample_id}")

                    cached_models = self._cached_trajectories[sample_id]

                    # if we have a specific model requested, use it
                    if model_name is not None:
                        if model_name not in cached_models:
                            raise ValueError(
                                f"Cached trajectory not found for model {model_name} and sample {sample_id}"
                            )
                        cached_result = cached_models[model_name]
                    else:
                        # no specific model requested - must be single model case
                        if len(cached_models) != 1:
                            raise ValueError(
                                f"Expected single model in cache for sample {sample_id}, found {len(cached_models)}: {list(cached_models.keys())}"
                            )
                        # get the single model's result
                        cached_result = next(iter(cached_models.values()))
                        model_name = cached_result.model_name

                    trajectory = cached_result.trajectory
                    agent_id = cached_result.agent_id

                    # notify progress callback with model name
                    if self.progress_callback:
                        await self.progress_callback.agent_loading(sample_id, model_name=model_name)
                else:
                    target = self._create_target(llm_config)
                    target_result = await target.run(sample, progress_callback=self.progress_callback)
                    trajectory = target_result.trajectory
                    agent_id = target_result.agent_id
                    model_name = target_result.model_name

                if self.progress_callback:
                    await self.progress_callback.grading_started(sample_id)

                grade_result, submission = await self.grader.grade(sample, trajectory)

                if self.progress_callback:
                    passed = self._check_score_against_gate(grade_result.score)

                    await self.progress_callback.sample_completed(sample_id, passed=passed, score=grade_result.score)

                return SampleResult(
                    sample=sample,
                    submission=submission,
                    trajectory=trajectory,
                    agent_id=agent_id,
                    grade=grade_result,
                    model_name=model_name,
                )
            except Exception as e:
                if self.progress_callback:
                    await self.progress_callback.sample_error(sample_id, str(e))
                raise

    async def run(self) -> RunnerResult:
        """Run evaluation on all samples."""
        samples = list(
            load_jsonl(self.suite.dataset, max_samples=self.suite.max_samples, sample_tags=self.suite.sample_tags)
        )

        self.results = []

        async with anyio.create_task_group() as tg:
            for llm_config in self.model_configs:
                for sample in samples:

                    async def run_and_append(s, cfg):
                        try:
                            result = await self.run_sample(s, llm_config=cfg)
                            self.results.append(result)
                        except Exception as e:
                            if self.progress_callback:
                                await self.progress_callback.sample_error(s.id, str(e))

                    tg.start_soon(run_and_append, sample, llm_config)

        metrics = self._calculate_metrics()
        gates_passed = self._check_gates(metrics)

        config = {
            "target": self.suite.target.model_dump(),
            "grader": self.suite.grader.model_dump(),
            "gate": self.suite.gate.model_dump(),
        }

        return RunnerResult(
            suite=self.suite.name, config=config, results=self.results, metrics=metrics, gates_passed=gates_passed
        )

    def _calculate_metrics(self) -> Metrics:
        """Calculate aggregate metrics."""
        total = len(self.results)
        if total == 0:
            return Metrics(total=0, avg_score=0.0)

        scores = [r.grade.score for r in self.results]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        per_model = None
        if self.suite.target.model_configs:
            model_results = defaultdict(list)
            for result in self.results:
                model_results[result.model_name].append(result)

            per_model = []
            for model_name, results in model_results.items():
                model_scores = [r.grade.score for r in results]
                model_avg = sum(model_scores) / len(model_scores) if model_scores else 0.0
                passed = sum(1 for r in results if self._check_score_against_gate(r.grade.score))
                failed = len(results) - passed

                per_model.append(
                    ModelMetrics(
                        model_name=model_name,
                        total=len(results),
                        avg_score=model_avg,
                        passed_samples=passed,
                        failed_samples=failed,
                    )
                )

        return Metrics(total=total, avg_score=avg_score, per_model=per_model)

    def _check_score_against_gate(self, score: float) -> bool:
        """Check if an individual score satisfies the gate."""
        return self.suite.gate.check_score(score)

    def _check_gates(self, metrics: Metrics) -> bool:
        """Check if gate is satisfied."""
        return self.suite.gate.check_score(metrics.avg_score)


async def run_suite(
    suite_path: Path,
    max_concurrent: int,
    progress_callback: Optional[ProgressCallback] = None,
    cached_results_path: Optional[Path] = None,
) -> RunnerResult:
    """Load and run a suite from YAML file."""
    with open(suite_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    suite = SuiteSpec.from_yaml(yaml_data, base_dir=suite_path.parent)

    cached_results = None
    if cached_results_path:
        if not cached_results_path.exists():
            raise ValueError(f"Cached results file not found: {cached_results_path}")

        with open(cached_results_path, "r") as f:
            cached_data = json.load(f)
            cached_results = RunnerResult(**cached_data)

        # validate that samples match by comparing IDs and inputs
        cached_sample_map = {result.sample.id: result.sample for result in cached_results.results}
        samples = list(load_jsonl(suite.dataset, max_samples=suite.max_samples, sample_tags=suite.sample_tags))

        for sample in samples:
            if sample.id not in cached_sample_map:
                raise ValueError(f"Sample ID {sample.id} not found in cached results")
            cached_sample = cached_sample_map[sample.id]
            if cached_sample.input != sample.input:
                raise ValueError(
                    f"Sample ID {sample.id} input mismatch: dataset has '{sample.input}' but cache has '{cached_sample.input}'"
                )

    runner = Runner(
        suite, max_concurrent=max_concurrent, progress_callback=progress_callback, cached_results=cached_results
    )
    return await runner.run()
