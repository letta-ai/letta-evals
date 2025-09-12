import asyncio
from pathlib import Path
from typing import List, Optional

import yaml

from letta_evals.datasets.loader import load_jsonl
from letta_evals.graders.base import Grader
from letta_evals.graders.rubric import RubricGrader
from letta_evals.graders.tool import ToolGrader
from letta_evals.models import Metrics, RunnerResult, Sample, SampleResult, SuiteSpec
from letta_evals.targets.agent import AgentTarget
from letta_evals.targets.base import Target
from letta_evals.types import GraderKind, ProgressCallback, TargetKind


class Runner:
    """Main evaluation runner."""

    def __init__(self, suite: SuiteSpec, max_concurrent: int, progress_callback: Optional[ProgressCallback] = None):
        self.suite: SuiteSpec = suite
        self.target: Target = self._create_target()
        self.grader: Grader = self._create_grader()
        self.results: List[SampleResult] = []
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.progress_callback = progress_callback

    def _create_target(self) -> Target:
        """Create target from spec."""
        if self.suite.target.kind == TargetKind.AGENT:
            return AgentTarget(
                base_url=self.suite.target.base_url,
                agent_id=self.suite.target.agent_id,
                agent_file=self.suite.target.agent_file,
                agent_script=self.suite.target.agent_script,
                api_key=self.suite.target.api_key,
                timeout=self.suite.target.timeout,
                base_dir=self.suite.target.base_dir,
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

    async def run_sample(self, sample: Sample, sample_id: int) -> SampleResult:
        """Run a single sample through target and grader."""
        async with self.semaphore:
            try:
                if self.progress_callback:
                    await self.progress_callback.sample_started(sample_id)

                target_result = await self.target.run(
                    sample, progress_callback=self.progress_callback, sample_id=sample_id
                )

                if self.progress_callback:
                    await self.progress_callback.grading_started(sample_id)

                grade_result = await self.grader.grade(sample, target_result.trajectory)

                if self.progress_callback:
                    passed = self._check_score_against_gate(grade_result.score)

                    await self.progress_callback.sample_completed(sample_id, passed=passed, score=grade_result.score)

                return SampleResult(
                    sample=sample,
                    trajectory=target_result.trajectory,
                    agent_id=target_result.agent_id,
                    grade=grade_result,
                    metadata=target_result.metadata,
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

        tasks = []
        for i, sample in enumerate(samples):
            tasks.append(self.run_sample(sample, sample_id=i))

        self.results = await asyncio.gather(*tasks)
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

        return Metrics(total=total, avg_score=avg_score)

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
