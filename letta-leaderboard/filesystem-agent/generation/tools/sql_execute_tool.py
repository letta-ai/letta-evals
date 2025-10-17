"""
SQL execution tool for agent question generation.

This tool allows the agent to execute SQL queries against the SQLite database
to explore data, verify uniqueness, and get answers.
"""

import sqlite3
import time
from pathlib import Path
from typing import Dict, Any


EXECUTE_SQL_TOOL_DICT = {
    "name": "execute_sql",
    "description": "Execute a SQL query against the database",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The SQL query to execute"}},
        "required": ["query"],
    },
}


class SQLExecuteTool:
    """Tool for executing SQL queries against the database."""

    def __init__(self, db_path: Path):
        """Initialize with path to SQLite database."""
        self.db_path = db_path

    def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute a SQL query and return results.

        Args:
            query: SQL query to execute

        Returns:
            Dictionary with:
                - success: Whether query executed successfully
                - result: Query results (list of dicts for SELECT, affected rows for INSERT/UPDATE)
                - row_count: Number of rows returned/affected
                - execution_time_ms: Query execution time in milliseconds
                - error: Error message if any
        """
        start_time = time.time()

        try:
            # Open database in read-only mode
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()

            # Enforce read-only operations
            query_upper = query.strip().upper()
            write_operations = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]

            if any(query_upper.startswith(op) for op in write_operations):
                raise ValueError(f"Write operations are not allowed. Only SELECT queries are permitted.")

            # Execute query
            cursor.execute(query)

            # Handle SELECT queries
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]
            row_count = len(result)

            # Simplify single-value results
            if row_count == 1 and len(result[0]) == 1:
                result = list(result[0].values())[0]

            execution_time_ms = (time.time() - start_time) * 1000

            conn.close()

            return {
                "success": True,
                "result": result,
                "row_count": row_count,
                "execution_time_ms": execution_time_ms,
                "error": None,
            }

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000

            return {
                "success": False,
                "result": None,
                "row_count": 0,
                "execution_time_ms": execution_time_ms,
                "error": str(e),
            }

    def get_schema_info(self) -> str:
        """Get database schema information for the agent prompt."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = cursor.fetchall()

        schema_info = []
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            schema_info.append(f"\nTable: {table_name}")
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                is_pk = " (PRIMARY KEY)" if col[5] else ""
                schema_info.append(f"  - {col_name}: {col_type}{is_pk}")

        conn.close()
        return "\n".join(schema_info)

    def get_database_overview(self) -> Dict[str, Any]:
        """Get comprehensive database overview including schema and statistics."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        overview = {"schema": "", "statistics": {}, "sample_ids": {}}

        # Get CREATE TABLE statements
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        create_statements = cursor.fetchall()
        overview["schema"] = "\n\n".join([row["sql"] for row in create_statements])

        # Get table statistics and sample IDs
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table["name"]

            # Get row count
            cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
            count = cursor.fetchone()["count"]
            overview["statistics"][table_name] = {"row_count": count}

            # Get sample unique IDs based on table
            if table_name == "people":
                cursor.execute(f"SELECT person_id, full_name FROM {table_name} LIMIT 3")
                samples = cursor.fetchall()
                overview["sample_ids"][table_name] = [f"{row['person_id']} ({row['full_name']})" for row in samples]
            elif table_name == "bank_accounts":
                cursor.execute(f"SELECT account_id, account_no FROM {table_name} LIMIT 3")
                samples = cursor.fetchall()
                overview["sample_ids"][table_name] = [
                    f"{row['account_id']} (Account: {row['account_no']})" for row in samples
                ]
            elif table_name == "credit_cards":
                cursor.execute(f"SELECT card_id, number FROM {table_name} LIMIT 3")
                samples = cursor.fetchall()
                overview["sample_ids"][table_name] = [
                    f"{row['card_id']} (Card: {row['number'][:4]}...)" for row in samples
                ]
            elif table_name == "vehicles":
                cursor.execute(f"SELECT vehicle_id, license_plate FROM {table_name} LIMIT 3")
                samples = cursor.fetchall()
                overview["sample_ids"][table_name] = [
                    f"{row['vehicle_id']} (Plate: {row['license_plate']})" for row in samples
                ]
            else:
                # Generic ID sampling
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                id_col = next((col[1] for col in columns if col[1].endswith("_id")), None)
                if id_col:
                    cursor.execute(f"SELECT {id_col} FROM {table_name} LIMIT 3")
                    samples = cursor.fetchall()
                    overview["sample_ids"][table_name] = [row[0] for row in samples]

        conn.close()
        return overview
