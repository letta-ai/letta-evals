"""Display utilities for question generation output."""

from typing import Any, Dict


# ANSI color codes
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


class DisplayMixin:
    """Mixin providing display methods for QuestionGeneratorAgent.

    Expects the host class to have:
        - self.quiet: bool
        - self.config: Dict[str, Any]
    """

    def _print_separator(self, char: str = "─", length: int = None):
        """Print a separator line."""
        if self.quiet:
            return
        if length is None:
            length = self.config.get("separator_length", 80)
        print(f"{Colors.DIM}{char * length}{Colors.ENDC}")

    def _print_tool_call(self, tool_name: str, tool_input: Dict[str, Any]):
        """Print tool call information."""
        if self.quiet:
            return
        print(f"\n{Colors.CYAN}Tool: {tool_name}{Colors.ENDC}")
        if tool_name == "execute_sql":
            # Format SQL query nicely
            query = tool_input["query"]
            # Simple formatting - indent and clean up
            formatted_query = "\n   ".join(line.strip() for line in query.split("\n") if line.strip())
            print(f"   {Colors.BOLD}Query:{Colors.ENDC}\n   {Colors.DIM}{formatted_query}{Colors.ENDC}")
        elif tool_name == "register_question":
            print(f"   {Colors.BOLD}Question:{Colors.ENDC} {tool_input['question']}")
            if "sql_queries" in tool_input and tool_input["sql_queries"]:
                print(f"   {Colors.BOLD}SQL Queries:{Colors.ENDC} {len(tool_input['sql_queries'])} queries")
                for i, query_info in enumerate(tool_input["sql_queries"][:2]):  # Show first 2
                    print(f"     {i + 1}. {query_info.get('description', 'Query')}")
            if "answer" in tool_input:
                print(
                    f"   {Colors.BOLD}Answer:{Colors.ENDC} {tool_input['answer'][:100]}..."
                    if len(tool_input["answer"]) > 100
                    else f"   {Colors.BOLD}Answer:{Colors.ENDC} {tool_input['answer']}"
                )

    def _print_tool_result(self, tool_name: str, result: Dict[str, Any]):
        """Print tool result information."""
        if self.quiet:
            return
        if tool_name == "execute_sql":
            if result["success"]:
                # Format result based on type
                res = result["result"]
                truncate_rows = self.config.get("truncate_result_rows", 3)
                truncate_str_len = self.config.get("truncate_result_string_length", 200)
                if isinstance(res, list) and len(res) > truncate_rows * 2:
                    # Truncate long lists
                    print(
                        f"   {Colors.GREEN}Result: {res[:truncate_rows]} ... (showing {truncate_rows} of {len(res)} rows){Colors.ENDC}"
                    )
                elif isinstance(res, str) and len(res) > truncate_str_len:
                    # Truncate long strings
                    print(f"   {Colors.GREEN}Result: {res[:truncate_str_len]}...{Colors.ENDC}")
                else:
                    print(f"   {Colors.GREEN}Result: {res}{Colors.ENDC}")
                print(
                    f"   {Colors.DIM}Rows: {result['row_count']} | Time: {result['execution_time_ms']:.1f}ms{Colors.ENDC}"
                )
            else:
                print(f"   {Colors.RED}Error: {result['error']}{Colors.ENDC}")
        elif tool_name == "register_question":
            if result["success"]:
                print(f"   {Colors.GREEN}{result['message']}{Colors.ENDC}")
                print(f"   {Colors.BOLD}Answer: {result['answer']}{Colors.ENDC}")
            else:
                print(f"   {Colors.RED}{result['error']}{Colors.ENDC}")

    def _print_progress(self, question_num: int, total: int, iteration: int, max_iterations: int):
        """Print progress information."""
        if self.quiet:
            return
        print(f"\n{Colors.BLUE}Question {question_num}/{total} | Iteration {iteration}/{max_iterations}{Colors.ENDC}")

    def _print_token_usage(self, usage, session_tokens=None):
        """Print token usage from response."""
        if usage:
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens

            # Update both session and total tokens
            if session_tokens:
                session_tokens["input"] += input_tokens
                session_tokens["output"] += output_tokens
                if not self.quiet:
                    print(
                        f"   Tokens: {input_tokens} in, {output_tokens} out (Session: {session_tokens['input']:,} in, {session_tokens['output']:,} out)"
                    )

            self.total_tokens["input"] += input_tokens
            self.total_tokens["output"] += output_tokens

    def _format_tool_response(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Format tool response for the agent."""
        if tool_name == "execute_sql":
            if result["success"]:
                return f"SQL executed successfully.\nResult: {result['result']}\nRows: {result['row_count']}"
            else:
                return f"SQL error: {result['error']}"
        elif tool_name == "register_question":
            if result["success"]:
                return f"Question registered! {result['message']}"
            else:
                return f"Registration failed: {result['error']}"
        return str(result)
