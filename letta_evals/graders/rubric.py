import json
import os
from pathlib import Path
from typing import Optional, Tuple

from anthropic import AsyncAnthropic, transform_schema
from dotenv import load_dotenv
from google import genai
from openai import AsyncOpenAI
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field as PydanticField

from letta_evals.graders.base import Grader
from letta_evals.graders.prompt_utils import build_judge_prompt
from letta_evals.models import AgentState, GradeResult, Sample
from letta_evals.types import LLMProvider

load_dotenv()


class _JudgeResponse(PydanticBaseModel):
    """Shared schema for judge responses across all providers."""

    score: float = PydanticField(ge=0.0, le=1.0, description="Score between 0.0 and 1.0")
    rationale: str = PydanticField(description="Explanation of the grading decision")


class RubricGrader(Grader):
    """Grader that uses an LLM judge with custom rubric prompts.

    The rubric is sent verbatim to the judge after template substitution.
    The framework only enforces the output schema ({score, rationale}) — it
    does not add a system prompt, wrap the rubric, or auto-append the
    question / ground truth / submission. Use template placeholders
    ({input}, {ground_truth}, {submission}, plus keys from
    ``sample.rubric_vars``) to reference those values from inside the rubric.
    """

    def __init__(
        self,
        prompt: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        provider: LLMProvider = LLMProvider.OPENAI,
        max_retries: int = 5,
        timeout: float = 120.0,
        extractor: str = "last_assistant",
        extractor_config: Optional[dict] = None,
        base_dir: Optional[Path] = None,
        system_prompt: Optional[str] = None,
    ):
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.provider = provider
        self.extractor_name = extractor
        self.base_dir = base_dir
        self._init_extractor(extractor, extractor_config, base_dir=base_dir)

        if provider == LLMProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")

            client_kwargs = {
                "api_key": api_key,
                "max_retries": max_retries,
                "timeout": timeout,
            }

            base_url = os.getenv("OPENAI_BASE_URL")
            if base_url:
                client_kwargs["base_url"] = base_url

            self.client = AsyncOpenAI(**client_kwargs)
        elif provider == LLMProvider.ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

            client_kwargs = {
                "api_key": api_key,
                "max_retries": max_retries,
                "timeout": timeout,
            }

            base_url = os.getenv("ANTHROPIC_BASE_URL")
            if base_url:
                client_kwargs["base_url"] = base_url

            self.client = AsyncAnthropic(**client_kwargs)
        elif provider == LLMProvider.GOOGLE:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables")
            self.client = genai.Client(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def grade(
        self, sample: Sample, trajectory, agent_state: Optional[AgentState] = None
    ) -> Tuple[GradeResult, str]:
        """Grade using LLM judge with rubric."""
        submission, extraction_time, early = self.extract(trajectory, agent_state)
        if early:
            return early

        # Per-sample rubric overrides the grader-level rubric.
        rubric_text = sample.rubric if sample.rubric is not None else self.prompt
        if rubric_text is None:
            return GradeResult(
                score=0.0,
                rationale=(
                    "No rubric available: neither the grader (prompt/prompt_path) "
                    "nor the sample (rubric/rubric_path) provided one."
                ),
                metadata={"error": "missing_rubric", "extraction_time": extraction_time},
            ), submission

        judge_prompt = build_judge_prompt(rubric_text, sample, submission)

        temperature = self.temperature

        try:
            if self.provider == LLMProvider.OPENAI:
                if (
                    self.model.startswith("o1") or self.model.startswith("o3") or "gpt-5" in self.model.lower()
                ) and temperature == 0.0:
                    temperature = 1.0

                messages = []
                if self.system_prompt is not None:
                    messages.append({"role": "system", "content": self.system_prompt})
                messages.append({"role": "user", "content": judge_prompt})

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "JudgeResponse",
                            "schema": _JudgeResponse.model_json_schema(),
                            "strict": True,
                        },
                    },
                )

                result_json = json.loads(response.choices[0].message.content)
                usage = response.usage.model_dump() if response.usage else None

            elif self.provider == LLMProvider.ANTHROPIC:
                kwargs = dict(
                    model=self.model,
                    max_tokens=4096,
                    temperature=temperature,
                    messages=[{"role": "user", "content": judge_prompt}],
                    output_config={"format": {"type": "json_schema", "schema": transform_schema(_JudgeResponse)}},
                )
                if self.system_prompt is not None:
                    kwargs["system"] = [
                        {"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}
                    ]
                response = await self.client.messages.create(**kwargs)

                result_json = json.loads(response.content[0].text)
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
                    "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
                }
            elif self.provider == LLMProvider.GOOGLE:
                google_config_kwargs = dict(
                    response_mime_type="application/json",
                    response_schema=_JudgeResponse,
                    temperature=temperature,
                )
                if self.system_prompt is not None:
                    google_config_kwargs["system_instruction"] = self.system_prompt
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=judge_prompt,
                    config=genai.types.GenerateContentConfig(**google_config_kwargs),
                )
                result_json = json.loads(response.text)
                usage = None
                if response.usage_metadata:
                    usage = {
                        "prompt_tokens": response.usage_metadata.prompt_token_count,
                        "completion_tokens": response.usage_metadata.candidates_token_count,
                        "total_tokens": response.usage_metadata.total_token_count,
                        "cached_content_token_count": getattr(response.usage_metadata, "cached_content_token_count", 0),
                    }
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            score = result_json.get("score")
            if score is None:
                raise ValueError("Model did not return a score")

            score = float(score)
            score = max(0.0, min(1.0, score))

            return GradeResult(
                score=score,
                rationale=result_json.get("rationale", ""),
                metadata={"model": self.model, "usage": usage, "extraction_time": extraction_time},
            ), submission

        except Exception as e:
            return GradeResult(
                score=0.0,
                rationale=f"Error during grading: {str(e)}",
                metadata={"error": str(e), "extraction_time": extraction_time},
            ), submission
