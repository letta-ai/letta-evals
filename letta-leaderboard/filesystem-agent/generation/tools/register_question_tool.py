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
            "difficulty": {
                "type": "string",
                "description": "The difficulty of the question from the following options: easy, medium, hard",
            },
            # question types generated using GPT-5
            "question_type": {
                "type": "string",
                "description": "The type of question from the following options: factual (direct retrieval), compositional (multi-hop), comparision (relative evaluation), logical (counting, math, filtering), explanatory (why/how)",
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
            "difficulty",
            "question_type",
            "required_files",
        ],
    },
}


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
        difficulty: str,
        question_type: str,
        required_files: List[str],
    ) -> Dict[str, Any]:
        """
        Register a new question by executing multiple SQL queries and storing results.

        Args:
            question: The human-readable question
            sql_queries: List of dicts with 'description' and 'query' for each SQL query
            answer: The direct natural language answer based on query results
            answer_reasoning: The reasoning process that led to the answer
            difficulty: The difficulty of the question from the following options: easy, medium, hard
            question_type: The type of question
            required_files: The files that are required to answer the question from the list of available files

        Returns:
            Dictionary with registration status and the answer
        """
        try:
            # Execute all SQL queries and collect results
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

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

            # Store question, answer and query results
            question_data = {
                "question": question,
                "answer": answer,
                "difficulty": difficulty,
                "question_type": question_type,
                "required_files": required_files,
                "answer_reasoning": answer_reasoning,
                "sql_queries": query_results,
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
