"""
Question registration tool for agent question generation.

This tool allows the agent to register validated questions with answers.
"""
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


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
        answer: str
    ) -> Dict[str, Any]:
        """
        Register a new question by executing multiple SQL queries and storing results.
        
        Args:
            question: The human-readable question
            sql_queries: List of dicts with 'description' and 'query' for each SQL query
            answer: The natural language answer based on query results
            
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
                description = query_info.get('description', 'Query')
                query = query_info['query']
                
                try:
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    
                    # Convert result to appropriate format
                    if len(rows) == 0:
                        result = "No results"
                    elif len(rows) == 1 and len(rows[0]) == 1:
                        # Single value result
                        result = list(dict(rows[0]).values())[0]
                    else:
                        # Multiple rows or columns
                        result = [dict(row) for row in rows]
                    
                    query_results.append({
                        "description": description,
                        "query": query,
                        "result": result
                    })
                except Exception as e:
                    query_results.append({
                        "description": description,
                        "query": query,
                        "error": str(e)
                    })
            
            conn.close()
            
            self.registered_count += 1
            
            # Store only question and answer
            question_data = {
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().isoformat()
            }
            
            # Append to output file
            with open(self.output_path, 'a') as f:
                f.write(json.dumps(question_data) + '\n')
            
            return {
                "success": True,
                "message": f"Question registered successfully. Answer: {answer}",
                "answer": answer,
                "query_results": query_results,
                "total_questions": self.registered_count
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }