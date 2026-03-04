"""Regression tests for filesystem benchmark generation guardrails."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


register_question_tool = load_module(
    "register_question_tool",
    "letta-leaderboard/filesystem-agent/generation/tools/register_question_tool.py",
)
validate_questions = load_module(
    "validate_questions",
    "letta-leaderboard/filesystem-agent/generation/validate_questions.py",
)


def make_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE numbers (value INTEGER)")
    conn.executemany("INSERT INTO numbers(value) VALUES (?)", [(1,), (2,)])
    conn.commit()
    conn.close()
    return path


def make_tool(tmp_path: Path):
    db_path = make_db(tmp_path / "fixture.db")
    output_path = tmp_path / "generated" / "agent_generated_questions.jsonl"
    return register_question_tool.RegisterQuestionTool(output_path=output_path, db_path=db_path)


def base_register_kwargs():
    return {
        "question": "Test question?",
        "sql_queries": [
            {"description": "q1", "query": "SELECT 1 AS value"},
            {"description": "q2", "query": "SELECT 1 AS value"},
            {"description": "q3", "query": "SELECT 1 AS value"},
        ],
        "answer": "1",
        "answer_reasoning": "Reasoning",
        "question_type": "aggregation",
        "required_files": ["a.txt", "b.txt", "c.txt"],
    }


def test_register_question_rejects_answer_mismatch(tmp_path):
    tool = make_tool(tmp_path)
    kwargs = base_register_kwargs()
    kwargs["answer"] = "2"

    result = tool.register(
        **kwargs,
        verification_query="SELECT 1 AS answer",
    )

    assert result["success"] is False
    assert "ANSWER MISMATCH" in result["error"]


def test_register_question_rejects_non_unique_verification_query(tmp_path):
    tool = make_tool(tmp_path)

    result = tool.register(
        **base_register_kwargs(),
        verification_query="SELECT value FROM numbers",
    )

    assert result["success"] is False
    assert "expected exactly 1" in result["error"]


def test_validate_questions_flags_address_join_multiplicity_risk():
    risky = """
    SELECT p.person_id
    FROM people p
    JOIN addresses a ON a.owner_id = p.person_id
    JOIN credit_cards c ON c.owner_id = p.person_id
    WHERE a.state = 'Idaho'
    GROUP BY p.person_id
    ORDER BY COUNT(c.card_id) DESC
    LIMIT 1
    """
    safe = """
    SELECT p.person_id
    FROM people p
    JOIN credit_cards c ON c.owner_id = p.person_id
    WHERE p.person_id IN (
        SELECT DISTINCT owner_id
        FROM addresses
        WHERE state = 'Idaho'
    )
    GROUP BY p.person_id
    ORDER BY COUNT(c.card_id) DESC
    LIMIT 1
    """

    assert validate_questions.has_address_join_multiplicity_risk(risky) is True
    assert validate_questions.has_address_join_multiplicity_risk(safe) is False


def test_validate_gt_against_db_checks_all_questions_when_sample_size_zero(tmp_path):
    db_path = make_db(tmp_path / "fixture.db")
    questions = [
        {"verification_query": "SELECT 1 AS answer", "answer": "1"},
        {"verification_query": "SELECT 2 AS answer", "answer": "3"},
    ]

    issues = validate_questions.validate_gt_against_db(questions, str(db_path), sample_size=0)

    assert len(issues) == 1
    assert issues[0]["id"] == 1
    assert issues[0]["flags"] == ["value_mismatch"]
