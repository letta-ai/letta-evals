"""
Pydantic models for LLM-generated questions and structured outputs.
"""

from typing import List, Literal

from pydantic import BaseModel


class QuestionAnswer(BaseModel):
    question: str
    answer: str
    difficulty: Literal["easy", "medium", "hard"]
    question_type: Literal["single_hop", "multi_hop", "comparison"]
    required_files: List[str]  # Files needed to answer the question
    reasoning_steps: List[str]  # Step-by-step reasoning required


class QuestionSet(BaseModel):
    questions: List[QuestionAnswer]
