from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import List

import anyio
from rich.console import Console

from letta_evals.visualization.rich_progress import DisplayMode, EvalProgress


@dataclass
class BenchmarkConfig:
    samples: int = 200
    workers: int = 20
    duration: float = 10.0
    refresh_fps: float = 4.0
    models: int = 2
    show_samples: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark rich progress render behavior")
    parser.add_argument("--samples", type=int, default=200, help="Number of synthetic samples")
    parser.add_argument("--workers", type=int, default=20, help="Concurrent event workers")
    parser.add_argument("--duration", type=float, default=10.0, help="Benchmark duration in seconds")
    parser.add_argument("--refresh-fps", type=float, default=4.0, help="UI render FPS cap")
    parser.add_argument("--models", type=int, default=2, help="Number of synthetic model names")
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Render sample table (off by default for very high-throughput runs)",
    )
    return parser


def parse_args(argv: List[str] | None = None) -> BenchmarkConfig:
    ns = build_parser().parse_args(argv)
    return BenchmarkConfig(
        samples=ns.samples,
        workers=ns.workers,
        duration=ns.duration,
        refresh_fps=ns.refresh_fps,
        models=ns.models,
        show_samples=ns.show_samples,
    )


async def _event_worker(
    progress: EvalProgress,
    total_samples: int,
    model_names: List[str],
    worker_id: int,
) -> None:
    rng = random.Random(worker_id)

    while True:
        sample_id = rng.randrange(total_samples)
        model_name = model_names[worker_id % len(model_names)]
        agent_id = f"agent-{sample_id % 10}"

        roll = rng.random()
        if roll < 0.10:
            await progress.sample_started(sample_id, agent_id=agent_id, model_name=model_name)
        elif roll < 0.45:
            total_messages = rng.randint(1, 8)
            message_num = rng.randint(1, total_messages)
            await progress.message_sending(
                sample_id,
                message_num,
                total_messages,
                agent_id=agent_id,
                model_name=model_name,
            )
        elif roll < 0.75:
            total_turns = rng.randint(2, 12)
            turn_num = rng.randrange(total_turns)
            await progress.turn_graded(
                sample_id=sample_id,
                turn_num=turn_num,
                total_turns=total_turns,
                turn_score=rng.random(),
                grader_key="quality",
                agent_id=agent_id,
                model_name=model_name,
            )
        else:
            await progress.grading_started(sample_id, agent_id=agent_id, model_name=model_name)

        # Yield frequently to maximize interleaving pressure.
        await anyio.sleep(0)


async def run_benchmark(config: BenchmarkConfig) -> dict[str, object]:
    model_names = [f"model-{idx + 1}" for idx in range(max(1, config.models))]
    total_evaluations = config.samples * len(model_names)
    progress = EvalProgress(
        suite_name="rich-progress-benchmark",
        total_samples=total_evaluations,
        target_kind="benchmark",
        grader_kind="synthetic",
        max_concurrent=config.workers,
        display_mode=DisplayMode.STANDARD,
        console=Console(),
        update_freq=config.refresh_fps,
        show_samples=config.show_samples,
        cached_mode=False,
        metric_labels={"quality": "Quality"},
    )

    await progress.start()
    try:
        async with anyio.create_task_group() as tg:
            for worker_id in range(config.workers):
                tg.start_soon(
                    _event_worker,
                    progress,
                    config.samples,
                    model_names,
                    worker_id,
                )
            await anyio.sleep(config.duration)
            tg.cancel_scope.cancel()

        # Mark each synthetic sample/model as completed once so counters remain bounded.
        for sample_id in range(config.samples):
            for model_name in model_names:
                await progress.sample_completed(
                    sample_id=sample_id,
                    agent_id=f"agent-{sample_id % 10}",
                    score=0.5,
                    model_name=model_name,
                    metric_scores={"quality": 0.5},
                    rationale="benchmark",
                    metric_rationales={"quality": "benchmark"},
                )
    finally:
        progress.stop()

    return progress.get_perf_snapshot()


def main(argv: List[str] | None = None) -> None:
    config = parse_args(argv)
    snapshot = anyio.run(run_benchmark, config)
    print(json.dumps(snapshot, indent=2, sort_keys=True))

