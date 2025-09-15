import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

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

    def __init__(self, suite: SuiteSpec, max_concurrent: int, progress_callback: Optional[ProgressCallback] = None):
        self.suite: SuiteSpec = suite
        self.grader: Grader = self._create_grader()
        self.results: List[SampleResult] = []
        self.max_concurrent = max_concurrent
        self.semaphore = anyio.Semaphore(max_concurrent)
        self.progress_callback = progress_callback
        self.model_configs = self._load_model_configs()

    def _load_model_configs(self) -> List[Optional[LlmConfig]]:
        """Load model configurations if specified."""
        if not self.suite.target.model_configs:
            return [None]  # no model configs, use default

        configs = []
        model_configs_dir = Path(__file__).parent.parent / "llm_model_configs"

        for config_name in self.suite.target.model_configs:
            config_path = model_configs_dir / f"{config_name}.json"
            if not config_path.exists():
                raise ValueError(f"Model config not found: {config_name}")

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

    async def run_sample(self, sample: Sample, sample_id: int, llm_config: Optional[LlmConfig] = None) -> SampleResult:
        """Run a single sample through target and grader."""
        async with self.semaphore:
            try:
                model_name = llm_config.model if llm_config else None
                if self.progress_callback:
                    await self.progress_callback.sample_started(sample_id, model_name=model_name)

                target = self._create_target(llm_config)
                target_result = await target.run(sample, progress_callback=self.progress_callback, sample_id=sample_id)

                if self.progress_callback:
                    await self.progress_callback.grading_started(sample_id)

                grade_result, submission = await self.grader.grade(sample, target_result.trajectory)

                if self.progress_callback:
                    passed = self._check_score_against_gate(grade_result.score)

                    await self.progress_callback.sample_completed(sample_id, passed=passed, score=grade_result.score)

                return SampleResult(
                    sample=sample,
                    submission=submission,
                    trajectory=target_result.trajectory,
                    agent_id=target_result.agent_id,
                    grade=grade_result,
                    model_name=target_result.model_name,
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
        sample_id = 0

        async with anyio.create_task_group() as tg:
            for llm_config in self.model_configs:
                for sample in samples:

                    async def run_and_append(sid, s, cfg):
                        result = await self.run_sample(s, sample_id=sid, llm_config=cfg)
                        self.results.append(result)

                    tg.start_soon(run_and_append, sample_id, sample, llm_config)
                    sample_id += 1

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

        # calculate per-model metrics if multiple models
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
    suite_path: Path, max_concurrent: int, progress_callback: Optional[ProgressCallback] = None
) -> RunnerResult:
    """Load and run a suite from YAML file."""
    with open(suite_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    suite = SuiteSpec.from_yaml(yaml_data, base_dir=suite_path.parent)

    runner = Runner(suite, max_concurrent=max_concurrent, progress_callback=progress_callback)
    return await runner.run()
