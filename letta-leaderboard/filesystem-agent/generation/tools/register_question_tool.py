"""
Question registration tool for agent question generation.

This tool allows the agent to register validated questions with answers.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REGISTER_QUESTION_TOOL_DICT = {
    "name": "register_question",
    "description": "Register a natural, investigative question that someone might genuinely ask about this population. Use multiple SQL queries to gather evidence and synthesize a comprehensive answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "A natural, conversational question someone might ask. Should feel investigative and genuine, not like a database query. Good questions often: compare groups, find outliers, investigate patterns, or express curiosity about relationships.",
            },
            "sql_queries": {
                "type": "array",
                "description": "Array of SQL queries that together help answer the question. Each query should explore a different aspect.",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "What this query investigates (e.g., 'Find all people with 5+ credit cards', 'Check their insurance coverage')",
                        },
                        "query": {"type": "string", "description": "The SQL query"},
                    },
                    "required": ["description", "query"],
                },
            },
            "answer": {
                "type": "string",
                "description": "Natural language answer synthesized from all query results. The answer MUST be a simple value: a number, a name, a date, etc. (e.g., 'John Smith', '5 pets', 'The rabbit is named Charlie').",
            },
            "answer_reasoning": {
                "type": "string",
                "description": "The reasoning process that led to the answer. This should be a thorough justification/explanation of the steps taken to arrive at the answer.",
            },
            # question types generated using GPT-5
            "question_type": {
                "type": "string",
                "description": "The type of question. Must be one of: multi_hop_chain, aggregation, set_intersection, negation, comparison_tiebreak, multi_entity_comparison, cross_file_counting, temporal_reasoning",
            },
            "verification_query": {
                "type": "string",
                "description": """A single SQL query that computes the answer END-TO-END using nested subqueries. 
MUST NOT contain hardcoded person IDs (pers-XXXX) or CASE statements with hardcoded answers.
The query must derive the answer from scratch starting from the original question parameters.

WRONG (hardcoded ID): SELECT full_name FROM people WHERE person_id = 'pers-0042'

CORRECT (end-to-end): 
SELECT p.full_name FROM people p WHERE p.person_id = (
    SELECT p2.person_id FROM people p2 
    JOIN addresses a ON a.owner_id = p2.person_id
    JOIN credit_cards c ON c.owner_id = p2.person_id
    WHERE a.state = (SELECT a2.state FROM pets pet JOIN addresses a2 ON a2.owner_id = pet.owner_id WHERE pet.name = 'Dawn' LIMIT 1)
    GROUP BY p2.person_id ORDER BY COUNT(c.card_id) DESC LIMIT 1
)""",
            },
            "required_files": {
                "type": "array",
                "description": "The files that are required to answer the question from the list of available files.",
                "items": {"type": "string", "description": "The name of the file"},
            },
        },
        "required": [
            "question",
            "sql_queries",
            "answer",
            "answer_reasoning",
            "question_type",
            "required_files",
            "verification_query",
        ],
    },
}


def compute_difficulty(question_type: str, required_files: List[str], sql_queries: List[Dict[str, str]]) -> str:
    """Derive difficulty from objective signals instead of LLM self-report.

    Scoring:
      - Files: 3 → 0pts, 4 → 1pt, 5+ → 2pts
      - SQL queries: 3 → 0pts, 4 → 1pt, 5+ → 2pts
      - Two-chain patterns (hardest): +2pts
        (multi_hop_chain, multi_entity_comparison)

    Total: 0-1 → easy, 2-3 → medium, 4+ → hard

    Based on empirical results:
      - multi_hop_chain: 40% Opus accuracy (HARD)
      - multi_entity_comparison: 50% Opus accuracy (HARD)
      - All other types: ~100% Opus accuracy (EASY)
    """
    score = 0

    num_files = len(required_files)
    if num_files == 4:
        score += 1
    elif num_files >= 5:
        score += 2

    num_queries = len(sql_queries)
    if num_queries == 4:
        score += 1
    elif num_queries >= 5:
        score += 2

    # Two-chain patterns are empirically the hardest
    hard_types = {"multi_hop_chain", "multi_entity_comparison"}
    if question_type in hard_types:
        score += 2  # +2 because these are significantly harder

    if score <= 1:
        return "easy"
    elif score <= 3:
        return "medium"
    else:
        return "hard"


class RegisterQuestionTool:
    """Tool for registering validated questions."""

    def __init__(self, output_path: Path, db_path: Path):
        """Initialize with output path for questions and database path."""
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.registered_count = 0

    def register(
        self,
        question: str,
        sql_queries: List[Dict[str, str]],
        answer: str,
        answer_reasoning: str,
        question_type: str,
        required_files: List[str],
        verification_query: str = "",
    ) -> Dict[str, Any]:
        """
        Register a new question by executing multiple SQL queries and storing results.

        Args:
            question: The human-readable question
            sql_queries: List of dicts with 'description' and 'query' for each SQL query
            answer: The direct natural language answer based on query results
            answer_reasoning: The reasoning process that led to the answer
            question_type: The type of question
            required_files: The files that are required to answer the question from the list of available files
            verification_query: A single SQL query that returns exactly 1 row with the answer

        Returns:
            Dictionary with registration status and the answer
        """
        # --- Guardrails ---

        # Check minimum files
        if len(required_files) < 3:
            return {
                "success": False,
                "error": f"Question must require at least 3 files, got {len(required_files)}: {required_files}. "
                "Make the question harder by involving more files.",
            }

        # Check minimum SQL queries
        if len(sql_queries) < 3:
            return {
                "success": False,
                "error": f"Must provide at least 3 SQL queries showing the reasoning chain, got {len(sql_queries)}. "
                "Add more queries to demonstrate the multi-step reasoning.",
            }

        # Check answer is not a negation/absence
        negation_phrases = [
            "does not own",
            "do not own",
            "doesn't own",
            "don't own",
            "no record",
            "no pets",
            "no vehicles",
            "no bank",
            "no credit",
            "no insurance",
            "not found",
            "none",
        ]
        answer_lower = answer.lower().strip()
        if any(phrase in answer_lower for phrase in negation_phrases):
            return {
                "success": False,
                "error": f"Answer must be a concrete value, not a negation/absence: '{answer}'. "
                "Rephrase the question so the answer is a name, number, or date.",
            }

        # Check answer is not question text (must be a simple value)
        question_phrases = ["the person with", "among", "residents of", "who has", "who owns"]
        if any(phrase in answer_lower for phrase in question_phrases):
            return {
                "success": False,
                "error": f"Answer contains question text: '{answer}'. "
                "The answer must be a simple value (name, number, date), not a description.",
            }

        # Check answer is reasonably short (names/numbers, not sentences)
        if len(answer) > 100:
            return {
                "success": False,
                "error": f"Answer is too long ({len(answer)} chars): '{answer[:50]}...'. "
                "The answer must be a simple value (name, number, date).",
            }

        # Check valid question type
        valid_types = [
            "multi_hop_chain",
            "aggregation",
            "set_intersection",
            "negation",
            "comparison_tiebreak",
            "multi_entity_comparison",
            "cross_file_counting",
            "temporal_reasoning",
        ]
        if question_type not in valid_types:
            return {
                "success": False,
                "error": f"Invalid question_type '{question_type}'. Must be one of: {valid_types}",
            }

        try:
            # Execute all SQL queries and collect results
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # --- Verification query check ---
            if not verification_query or not verification_query.strip():
                conn.close()
                return {
                    "success": False,
                    "error": "MISSING VERIFICATION QUERY: You must provide a verification_query that computes "
                    "the answer end-to-end using nested subqueries. This is required to validate correctness.",
                }

            # REJECT if verification query contains hardcoded person IDs
            if "pers-" in verification_query:
                conn.close()
                return {
                    "success": False,
                    "error": "INVALID VERIFICATION QUERY: Contains hardcoded person ID (pers-XXXX). "
                    "The verification query must compute the answer end-to-end using nested subqueries, "
                    "not look up a pre-determined person ID. Rewrite the query to derive the person from scratch.",
                }

            # REJECT if verification query contains CASE with hardcoded names
            if "CASE" in verification_query.upper() and "THEN '" in verification_query:
                conn.close()
                return {
                    "success": False,
                    "error": "INVALID VERIFICATION QUERY: Contains CASE statement with hardcoded answer. "
                    "The verification query must compute and return the answer, not embed it in a CASE statement.",
                }

                try:
                    cursor.execute(verification_query)
                    verification_rows = cursor.fetchall()
                    if len(verification_rows) != 1:
                        conn.close()
                        return {
                            "success": False,
                            "error": f"Verification query returned {len(verification_rows)} rows, expected exactly 1. "
                            "The answer is not unique — refine the question conditions.",
                        }
                    # Check that the verification query result matches the provided answer
                    verification_value = str(list(dict(verification_rows[0]).values())[0])
                    answer_normalized = answer.strip()
                    # For numeric answers, also try to match formatted versions
                    if verification_value != answer_normalized:
                        # Try removing $ and commas for currency
                        answer_cleaned = answer_normalized.replace("$", "").replace(",", "")
                        verification_cleaned = verification_value.replace("$", "").replace(",", "")
                        # Try to compare as numbers if possible
                        try:
                            if abs(float(answer_cleaned) - float(verification_cleaned)) < 0.01:
                                pass  # Close enough for floats
                            else:
                                conn.close()
                                return {
                                    "success": False,
                                    "error": f"ANSWER MISMATCH: Verification query returned '{verification_value}' "
                                    f"but answer field says '{answer}'. The generator's reasoning is wrong. "
                                    "Re-run the SQL queries and fix the answer.",
                                }
                        except (ValueError, TypeError):
                            # Not numeric, do string comparison
                            if verification_value.lower() != answer_normalized.lower():
                                conn.close()
                                return {
                                    "success": False,
                                    "error": f"ANSWER MISMATCH: Verification query returned '{verification_value}' "
                                    f"but answer field says '{answer}'. The generator's reasoning is wrong. "
                                    "Re-run the SQL queries and fix the answer.",
                                }
                except Exception as e:
                    conn.close()
                    return {
                        "success": False,
                        "error": f"Verification query failed: {str(e)}. Fix the query and try again.",
                    }

            query_results = []
            for query_info in sql_queries:
                description = query_info.get("description", "Query")
                query = query_info["query"]

                try:
                    cursor.execute(query)
                    rows = cursor.fetchall()

                    # Convert result to appropriate format
                    if len(rows) == 0:
                        result = "No results"
                        raise Exception("No results found for the query")
                    elif len(rows) == 1 and len(rows[0]) == 1:
                        # Single value result
                        result = list(dict(rows[0]).values())[0]
                    else:
                        # Multiple rows or columns
                        result = [dict(row) for row in rows]

                    query_results.append({"description": description, "query": query, "result": result})
                except Exception as e:
                    query_results.append({"description": description, "query": query, "error": str(e)})

            conn.close()

            self.registered_count += 1

            # Compute difficulty from objective signals
            difficulty = compute_difficulty(question_type, required_files, sql_queries)

            # Store question, answer and query results
            question_data = {
                "question": question,
                "answer": answer,
                "difficulty": difficulty,
                "question_type": question_type,
                "required_files": required_files,
                "answer_reasoning": answer_reasoning,
                "sql_queries": query_results,
                "verification_query": verification_query,
                "timestamp": datetime.now().isoformat(),
            }

            # Append to output file
            with open(self.output_path, "a") as f:
                f.write(json.dumps(question_data) + "\n")

            # letta-evals format
            question_data_formatted = {
                "input": question,
                "ground_truth": answer,
                "agent_args": {
                    "tags": [],
                    "extra": {
                        "required_files": required_files,
                        "question_type": question_type,
                        "difficulty": difficulty,
                    },
                },
            }
            with open(self.output_path.parent / "agent_generated_questions_parsed.jsonl", "a") as f:
                f.write(json.dumps(question_data_formatted) + "\n")

            return {
                "success": True,
                "message": f"Question registered successfully. Answer: {answer}",
                "answer": answer,
                "query_results": query_results,
                "total_questions": self.registered_count,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
