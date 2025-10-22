import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from letta_client import AgentState, LettaMessageUnion
from openai import AsyncOpenAI

from letta_evals.extractors import extractor_requires_agent_state, get_extractor
from letta_evals.graders.base import Grader
from letta_evals.graders.prompt_utils import JUDGE_SYSTEM_PROMPT, build_judge_prompt
from letta_evals.models import GradeResult, Sample
from letta_evals.types import LLMProvider

load_dotenv()


class RubricGrader(Grader):
    """Grader that uses an LLM judge with custom rubric prompts."""

    def __init__(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        provider: LLMProvider = LLMProvider.OPENAI,
        max_retries: int = 5,
        timeout: float = 120.0,
        extractor: str = "last_assistant",
        extractor_config: Optional[dict] = None,
        base_dir: Optional[Path] = None,
        rubric_vars: Optional[List[str]] = None,
    ):
        self.prompt = prompt
        self.model = model
        self.temperature = temperature
        self.provider = provider
        self.extractor_name = extractor
        self.base_dir = base_dir
        self.rubric_vars = rubric_vars or []
        self.extractor = get_extractor(extractor, extractor_config, base_dir=base_dir)
        self._requires_agent_state = extractor_requires_agent_state(extractor, base_dir=base_dir)

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
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @property
    def requires_agent_state(self) -> bool:
        """Whether this grader's extractor requires agent_state."""
        return self._requires_agent_state

    async def grade(
        self, sample: Sample, trajectory: List[List[LettaMessageUnion]], agent_state: Optional[AgentState] = None
    ) -> Tuple[GradeResult, str]:
        """Grade using LLM judge with rubric."""
        submission = self.extractor(trajectory, agent_state=agent_state)

        judge_prompt = build_judge_prompt(self.prompt, sample, submission, self.rubric_vars)

        temperature = self.temperature

        try:
            if self.provider == LLMProvider.OPENAI:
                if (
                    self.model.startswith("o1") or self.model.startswith("o3") or "gpt-5" in self.model.lower()
                ) and temperature == 0.0:
                    temperature = 1.0

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": judge_prompt},
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )

                result_json = json.loads(response.choices[0].message.content)
                usage = response.usage.model_dump() if response.usage else None

            elif self.provider == LLMProvider.ANTHROPIC:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=temperature,
                    system=[{"type": "text", "text": JUDGE_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                    messages=[
                        {"role": "user", "content": judge_prompt},
                        {"role": "assistant", "content": "{"},  # prefill trick
                    ],
                )

                # extract text from response
                response_text = "{"
                for block in response.content:
                    if hasattr(block, "text"):
                        response_text += block.text

                result_json = json.loads(response_text)
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
                    "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
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
                metadata={"model": self.model, "usage": usage},
            ), submission

        except Exception as e:
            return GradeResult(
                score=0.0, rationale=f"Error during grading: {str(e)}", metadata={"error": str(e)}
            ), submission
