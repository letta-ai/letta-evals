import os
from functools import partial
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel

from letta_evals.extractors import get_extractor
from letta_evals.graders.base import Grader
from letta_evals.graders.prompt_utils import build_judge_prompt
from letta_evals.graders.scorers import ChecklistResponse, ScoreResponse, check_response_model, checklist, score
from letta_evals.models import GradeResult, ModelJudgeGraderSpec, Sample
from letta_evals.types import LLMProvider
from letta_evals.utils import load_object

# Total judge calls per sample when replies are rejected: the reply doesn't
# match the response format, the scorer raises ValueError, or there's no reply.
MAX_ATTEMPTS = 3


class RubricGrader(Grader):
    """Grader that uses an LLM judge with custom rubric prompts.

    The rubric is a Jinja template, rendered and sent verbatim to the judge.
    The judge replies in the grader's response format (a Pydantic model), and
    the scorer turns the reply into a score. The framework does not add a
    system prompt, wrap the rubric, or auto-append the question / ground truth
    / submission. Reference those values from inside the rubric with
    ``{{ input }}``, ``{{ ground_truth }}``, ``{{ submission }}`` and keys from
    ``sample.rubric_vars``.
    """

    def __init__(self, spec: ModelJudgeGraderSpec):
        self.spec = spec
        self.system_prompt = spec.system_prompt_path.read_text() if spec.system_prompt_path else None
        self.extractor = get_extractor(spec.extractor, spec.extractor_config, base_dir=spec.base_dir)

        if spec.scorer == "score":
            self.scorer, self.response_format = score, ScoreResponse
        elif spec.scorer == "checklist":
            self.scorer, self.response_format = partial(checklist, criteria_var=spec.criteria_var), ChecklistResponse
        else:
            self.scorer = load_object(spec.scorer, base_dir=spec.base_dir)
            if not callable(self.scorer):
                raise ValueError(f"scorer {spec.scorer} is not callable")
        if spec.response_format is not None:
            self.response_format = load_object(spec.response_format, base_dir=spec.base_dir)
        check_response_model(self.response_format)
        if spec.scorer == "score" and "score" not in self.response_format.model_fields:
            raise ValueError(f"scorer 'score' needs a score field, and {spec.response_format} has none")

        if spec.provider == LLMProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")

            from openai import AsyncOpenAI

            client_kwargs = {
                "api_key": api_key,
                "max_retries": spec.max_retries,
                "timeout": spec.timeout,
            }

            base_url = os.getenv("OPENAI_BASE_URL")
            if base_url:
                client_kwargs["base_url"] = base_url

            self.client = AsyncOpenAI(**client_kwargs)
        elif spec.provider == LLMProvider.ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

            from anthropic import AsyncAnthropic

            client_kwargs = {
                "api_key": api_key,
                "max_retries": spec.max_retries,
                "timeout": spec.timeout,
            }

            base_url = os.getenv("ANTHROPIC_BASE_URL")
            if base_url:
                client_kwargs["base_url"] = base_url

            self.client = AsyncAnthropic(**client_kwargs)
        else:
            raise ValueError(f"Unsupported provider: {spec.provider}")

    async def grade(
        self, sample: Sample, trajectory, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[GradeResult, str]:
        """Grade using LLM judge with rubric."""
        submission, extraction_time, early = self.extract(trajectory, context)
        if early:
            return early

        # Per-sample rubric overrides the grader-level rubric.
        rubric_text = sample.rubric if sample.rubric is not None else self.spec.prompt
        judge_prompt = build_judge_prompt(rubric_text, sample, submission)

        try:
            rejections = []
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    response, usage = await self._ask_judge(judge_prompt)
                    score = float(self.scorer(sample, response))
                    if not 0.0 <= score <= 1.0:
                        raise ValueError(f"scorer returned {score}, outside [0, 1]")
                except ValueError as e:  # includes pydantic's ValidationError
                    rejections.append(str(e))
                    continue

                rationale = getattr(response, "rationale", None)
                return GradeResult(
                    score=score,
                    rationale=rationale if isinstance(rationale, str) else response.model_dump_json(),
                    metadata={
                        "model": self.spec.model,
                        "response": response.model_dump(mode="json"),
                        "attempts": attempt,
                        "usage": usage,
                        "extraction_time": extraction_time,
                    },
                ), submission

            raise ValueError(f"Judge reply rejected {MAX_ATTEMPTS} times; last: {rejections[-1]}")

        except Exception as e:
            return GradeResult(
                score=0.0,
                rationale=f"Error during grading: {str(e)}",
                metadata={"error": str(e), "extraction_time": extraction_time},
            ), submission

    async def _ask_judge(self, judge_prompt: str) -> Tuple[BaseModel, Optional[Dict[str, Any]]]:
        """Call the judge once and return its parsed reply and token usage.

        Raises:
            ValueError: If the reply doesn't match the response format, or there is none.
        """
        if self.spec.provider == LLMProvider.OPENAI:
            kwargs = dict(
                model=self.spec.model,
                input=judge_prompt,
                text_format=self.response_format,
                max_output_tokens=self.spec.max_output_tokens,
                store=False,
            )
            if self.spec.reasoning_effort is not None:
                kwargs["reasoning"] = {"effort": self.spec.reasoning_effort}
            if self.system_prompt is not None:
                kwargs["instructions"] = self.system_prompt
            response = await self.client.responses.parse(**kwargs)

            usage = response.usage.model_dump() if response.usage else None
            if response.output_parsed is None:
                raise ValueError(f"Judge returned no parsable reply (status={response.status})")
            return response.output_parsed, usage

        kwargs = dict(
            model=self.spec.model,
            max_tokens=self.spec.max_output_tokens,
            messages=[{"role": "user", "content": judge_prompt}],
            output_format=self.response_format,
        )
        if self.spec.reasoning_effort is not None:
            kwargs["output_config"] = {"effort": self.spec.reasoning_effort}
        if self.system_prompt is not None:
            kwargs["system"] = [{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}]
        response = await self.client.messages.parse(**kwargs)

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
        }
        # Models that think by default put a thinking block before the text; parsed_output skips it.
        if response.parsed_output is None:
            raise ValueError(f"Judge returned no reply (stop_reason={response.stop_reason})")
        return response.parsed_output, usage
