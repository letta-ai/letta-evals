import json
import os
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field as PydanticField

from letta_evals.extractors import get_extractor
from letta_evals.graders.base import Grader
from letta_evals.graders.prompt_utils import build_judge_prompt
from letta_evals.models import GradeResult, ModelJudgeGraderSpec, Sample
from letta_evals.types import LLMProvider


class _JudgeResponse(PydanticBaseModel):
    """Shared schema for judge responses across all providers.

    Sent verbatim to both providers (OpenAI, Anthropic) as a
    JSON schema response format. The schema acts as guidance — bounds on
    ``score`` are re-applied Python-side via a post-parse clamp because
    not every provider honors numeric bound keywords (``minimum`` /
    ``maximum``) consistently.

    ``rationale`` comes first: providers emit fields in schema order, so a
    non-reasoning judge writes its reasoning before it commits to a score.
    """

    rationale: str = PydanticField(description="Explanation of the grading decision")
    score: float = PydanticField(ge=0.0, le=1.0, description="Score between 0.0 and 1.0")


class RubricGrader(Grader):
    """Grader that uses an LLM judge with custom rubric prompts.

    The rubric is sent verbatim to the judge after template substitution.
    The framework only enforces the output schema ({rationale, score}) — it
    does not add a system prompt, wrap the rubric, or auto-append the
    question / ground truth / submission. Use template placeholders
    ({input}, {ground_truth}, {submission}, plus keys from
    ``sample.rubric_vars``) to reference those values from inside the rubric.
    """

    def __init__(self, spec: ModelJudgeGraderSpec):
        self.spec = spec
        self.system_prompt = spec.system_prompt_path.read_text() if spec.system_prompt_path else None
        self.extractor = get_extractor(spec.extractor, spec.extractor_config, base_dir=spec.base_dir)

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
            if self.spec.provider == LLMProvider.OPENAI:
                kwargs = dict(
                    model=self.spec.model,
                    input=judge_prompt,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "JudgeResponse",
                            "schema": _JudgeResponse.model_json_schema(),
                            "strict": False,
                        }
                    },
                    max_output_tokens=self.spec.max_output_tokens,
                    store=False,
                )
                if self.spec.reasoning_effort is not None:
                    kwargs["reasoning"] = {"effort": self.spec.reasoning_effort}
                if self.system_prompt is not None:
                    kwargs["instructions"] = self.system_prompt
                response = await self.client.responses.create(**kwargs)

                result_json = json.loads(response.output_text)
                usage = response.usage.model_dump() if response.usage else None

            elif self.spec.provider == LLMProvider.ANTHROPIC:
                from anthropic import transform_schema

                output_config = {"format": {"type": "json_schema", "schema": transform_schema(_JudgeResponse)}}
                if self.spec.reasoning_effort is not None:
                    output_config["effort"] = self.spec.reasoning_effort
                kwargs = dict(
                    model=self.spec.model,
                    max_tokens=self.spec.max_output_tokens,
                    messages=[{"role": "user", "content": judge_prompt}],
                    output_config=output_config,
                )
                if self.system_prompt is not None:
                    kwargs["system"] = [
                        {"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}
                    ]
                response = await self.client.messages.create(**kwargs)

                # Models that think by default put a thinking block before the text.
                text = next((block.text for block in response.content if block.type == "text"), None)
                if text is None:
                    raise ValueError(f"Judge returned no text (stop_reason={response.stop_reason})")
                result_json = json.loads(text)
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
                    "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
                }
            else:
                raise ValueError(f"Unsupported provider: {self.spec.provider}")

            score = result_json.get("score")
            if score is None:
                raise ValueError("Model did not return a score")

            score = float(score)
            score = max(0.0, min(1.0, score))

            return GradeResult(
                score=score,
                rationale=result_json.get("rationale", ""),
                metadata={"model": self.spec.model, "usage": usage, "extraction_time": extraction_time},
            ), submission

        except Exception as e:
            return GradeResult(
                score=0.0,
                rationale=f"Error during grading: {str(e)}",
                metadata={"error": str(e), "extraction_time": extraction_time},
            ), submission
