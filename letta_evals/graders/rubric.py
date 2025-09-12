import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from letta_client import LettaMessageUnion
from openai import AsyncOpenAI

from letta_evals.extractors import get_extractor
from letta_evals.graders.base import Grader
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
        extractor: str = "last_assistant",
        extractor_config: Optional[dict] = None,
    ):
        self.prompt = prompt
        self.model = model
        self.temperature = temperature
        self.provider = provider
        self.extractor = get_extractor(extractor, extractor_config)

        if provider == LLMProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")

            client_kwargs = {"api_key": api_key}

            base_url = os.getenv("OPENAI_BASE_URL")
            if base_url:
                client_kwargs["base_url"] = base_url

            self.client = AsyncOpenAI(**client_kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def grade(self, sample: Sample, trajectory: List[List[LettaMessageUnion]]) -> GradeResult:
        """Grade using LLM judge with rubric."""
        submission = self.extractor(trajectory)

        judge_prompt = self._build_judge_prompt(sample, submission)

        temperature = self.temperature
        if (
            self.model.startswith("o1") or self.model.startswith("o3") or "gpt-5" in self.model.lower()
        ) and temperature == 0.0:
            temperature = 1.0

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )

            result_json = json.loads(response.choices[0].message.content)

            score = result_json.get("score")
            if score is None:
                raise ValueError("Model did not return a score")

            score = float(score)
            score = max(0.0, min(1.0, score))

            return GradeResult(
                score=score,
                rationale=result_json.get("rationale", ""),
                metadata={"model": self.model, "usage": response.usage.model_dump() if response.usage else None},
            )

        except Exception as e:
            return GradeResult(score=0.0, rationale=f"Error during grading: {str(e)}", metadata={"error": str(e)})

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the judge."""
        return """You are an evaluation judge. You will be given:
1. A rubric describing evaluation criteria
2. An input/question
3. A submission to evaluate

Evaluate the submission according to the rubric and return a JSON response with:
{
    "score": (REQUIRED: a decimal number between 0.0 and 1.0 inclusive),
    "rationale": "explanation of your grading decision"
}

IMPORTANT:
- The score MUST be a number between 0.0 and 1.0 (inclusive)
- 0.0 means complete failure, 1.0 means perfect
- Use decimal values for partial credit (e.g., 0.25, 0.5, 0.75)
- Be objective and follow the rubric strictly"""

    def _build_judge_prompt(self, sample: Sample, submission: str) -> str:
        """Build the prompt for the judge."""
        prompt = self.prompt

        prompt = prompt.replace("{input}", str(sample.input))
        prompt = prompt.replace("{submission}", submission)

        if sample.ground_truth:
            prompt = prompt.replace("{ground_truth}", sample.ground_truth)

        if "{metadata}" in prompt and sample.metadata:
            metadata_str = json.dumps(sample.metadata.model_dump(), indent=2)
            prompt = prompt.replace("{metadata}", metadata_str)

        parts = [
            "## Rubric",
            prompt,
            "",
            "## Input",
            str(sample.input),
        ]

        if sample.ground_truth:
            parts.extend(["", "## Ground Truth Answer", sample.ground_truth])

        parts.extend(
            [
                "",
                "## Submission to Evaluate",
                submission,
                "",
                "Please evaluate the submission according to the rubric and return your judgment in JSON format.",
            ]
        )

        return "\n".join(parts)
