"""Regression tests for filesystem benchmark generation guardrails."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    module_dir = str(path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
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
question_generator = load_module(
    "question_generator",
    "letta-leaderboard/filesystem-agent/generation/question_generator.py",
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
        verification_query="SELECT answer FROM (SELECT 1 AS answer)",
    )

    assert result["success"] is False
    assert "ANSWER MISMATCH" in result["error"]


def test_register_question_rejects_non_unique_verification_query(tmp_path):
    tool = make_tool(tmp_path)

    result = tool.register(
        **base_register_kwargs(),
        verification_query="SELECT value FROM (SELECT value FROM numbers)",
    )

    assert result["success"] is False
    assert "expected exactly 1" in result["error"]


def test_register_question_rejects_address_join_multiplicity_risk(tmp_path):
    tool = make_tool(tmp_path)

    result = tool.register(
        **base_register_kwargs(),
        verification_query="""
        SELECT person_id
        FROM (
            SELECT p.person_id
            FROM people p
            JOIN addresses a ON a.owner_id = p.person_id
            JOIN credit_cards c ON c.owner_id = p.person_id
            WHERE a.state = 'Idaho'
            GROUP BY p.person_id
            ORDER BY COUNT(c.card_id) DESC
            LIMIT 1
        )
        """,
    )

    assert result["success"] is False
    assert "Raw JOIN addresses aggregation" in result["error"]


def test_register_question_rejects_non_correct_dataset_audit(tmp_path, monkeypatch):
    tool = make_tool(tmp_path)
    monkeypatch.setattr(
        register_question_tool,
        "audit_dataset_row",
        lambda row, db_path: SimpleNamespace(status="ambiguous", valid_answers=["A", "B"], note="tie"),
    )

    result = tool.register(
        **base_register_kwargs(),
        verification_query="SELECT answer FROM (SELECT 1 AS answer)",
    )

    assert result["success"] is False
    assert "QUESTION FAILED DATASET AUDIT" in result["error"]
    assert "ambiguous" in result["error"]


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


def test_resolve_parsed_dataset_path_finds_sibling(tmp_path):
    raw_path = tmp_path / "agent_generated_questions.jsonl"
    parsed_path = tmp_path / "agent_generated_questions_parsed.jsonl"
    raw_path.write_text("{}\n")
    parsed_path.write_text("{}\n")

    assert validate_questions.resolve_parsed_dataset_path(raw_path) == parsed_path


def test_audit_parsed_dataset_converts_non_correct_results_to_issues(tmp_path, monkeypatch):
    parsed_path = tmp_path / "agent_generated_questions_parsed.jsonl"
    parsed_path.write_text("{}\n")
    db_path = tmp_path / "fixture.db"
    db_path.write_text("")

    monkeypatch.setattr(validate_questions, "load_dataset_rows", lambda path: [{"input": "q"}])
    monkeypatch.setattr(
        validate_questions,
        "audit_dataset_rows",
        lambda rows, db: [
            validate_questions.AuditResult(
                index=3,
                status="wrong",
                question_type="aggregation",
                ground_truth="A",
                valid_answers=["B"],
                question="Question?",
            )
        ],
    )
    monkeypatch.setattr(validate_questions, "summarize_audit_results", lambda results: {"wrong": 1})

    issues = validate_questions.audit_parsed_dataset(parsed_path, db_path)

    assert len(issues) == 1
    assert issues[0]["type"] == "parsed_dataset_audit"
    assert issues[0]["flags"] == ["wrong"]
    assert issues[0]["valid_answers"] == ["B"]


def test_run_validation_auto_audits_sibling_parsed_dataset(tmp_path, monkeypatch):
    raw_path = tmp_path / "agent_generated_questions.jsonl"
    parsed_path = tmp_path / "agent_generated_questions_parsed.jsonl"
    raw_question = {
        "question": "Question?",
        "answer": "A",
        "question_type": "aggregation",
        "verification_query": "SELECT 1",
    }
    raw_path.write_text(
        json.dumps(raw_question) + "\n"
    )
    parsed_path.write_text("{}\n")
    db_path = tmp_path / "fixture.db"
    db_path.write_text("")

    monkeypatch.setattr(validate_questions, "check_forbidden_terms", lambda questions: [])
    monkeypatch.setattr(validate_questions, "check_answer_quality", lambda questions: [])
    monkeypatch.setattr(validate_questions, "check_verification_query", lambda questions: [])
    monkeypatch.setattr(validate_questions, "validate_gt_against_db", lambda questions, db, sample_size=0: [])
    monkeypatch.setattr(
        validate_questions,
        "audit_parsed_dataset",
        lambda parsed_path, db_path: [{"id": 0, "type": "parsed_dataset_audit", "flags": ["wrong"]}],
    )

    issues = validate_questions.run_validation(raw_path, db_path=db_path)

    assert len(issues) == 1
    assert issues[0]["type"] == "parsed_dataset_audit"
    assert issues[0]["flags"] == ["wrong"]


def test_generate_questions_tops_up_until_target_count(tmp_path):
    agent = object.__new__(question_generator.QuestionGeneratorAgent)
    accepted = []
    outcomes = iter([False, False, True, True])
    trace = {}

    agent.config = {
        "max_retries_per_question": 2,
        "max_iterations_per_question": 1,
        "max_failed_questions_per_run": 3,
    }
    agent.model = "test-model"
    agent.output_path = tmp_path / "agent_generated_questions.jsonl"
    agent.total_tokens = {"input": 0, "output": 0}
    agent._build_type_schedule = lambda n: ["aggregation"] * n
    agent._print_separator = lambda *args, **kwargs: None
    agent._save_full_trace = (
        lambda session_id, conversations, target_count: trace.update(
            {"conversations": conversations, "target_count": target_count}
        )
    )
    agent.get_existing_questions = lambda: accepted.copy()

    def generate_single_question(question_number, total, existing_questions, max_iterations, question_type=None):
        success = next(outcomes)
        if success:
            accepted.append({"question": f"Q{len(accepted) + 1}", "answer": "A"})
        return success, [{"success": success, "slot": question_number, "type": question_type}]

    agent.generate_single_question = generate_single_question

    agent.generate_questions(num_questions=2, question_type="aggregation")

    assert len(accepted) == 2
    assert trace["target_count"] == 2
    assert len(trace["conversations"]) == 3
    assert trace["conversations"][0]["success"] is False
    assert trace["conversations"][-1]["success"] is True


def test_generate_questions_parallel_tops_up_until_target_count(tmp_path):
    agent = object.__new__(question_generator.QuestionGeneratorAgent)
    accepted = []
    outcomes = iter([False, False, True, True])

    agent.config = {
        "max_retries_per_question": 2,
        "max_iterations_per_question": 1,
        "max_failed_questions_per_run": 3,
    }
    agent.model = "test-model"
    agent.output_path = tmp_path / "agent_generated_questions.jsonl"
    agent.total_tokens = {"input": 0, "output": 0}
    agent.quiet = False
    agent._print_separator = lambda *args, **kwargs: None
    agent.get_existing_questions = lambda: accepted.copy()

    def generate_single_question(question_number, total, existing_questions, max_iterations, question_type=None):
        success = next(outcomes)
        if success:
            accepted.append({"question": f"Q{len(accepted) + 1}", "answer": "A"})
        return success, [{"success": success, "slot": question_number, "type": question_type}]

    agent.generate_single_question = generate_single_question

    agent.generate_questions_parallel(num_questions=2, num_workers=1, question_type="aggregation")

    assert len(accepted) == 2
