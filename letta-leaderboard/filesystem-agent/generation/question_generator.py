"""
Agent-based question generator for the file benchmark.

This script runs an AI agent that generates difficult questions by:
1. Exploring the SQLite database
2. Finding unique identifiers and relationships
3. Creating questions that require multiple file lookups
4. Verifying answers through SQL execution
"""

import argparse
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from anthropic import Anthropic
from context import ContextMixin
from display import Colors, DisplayMixin
from dotenv import load_dotenv
from jinja2 import Template
from parallel import ParallelMixin
from tools.register_question_tool import REGISTER_QUESTION_TOOL_DICT, RegisterQuestionTool
from tools.sql_execute_tool import EXECUTE_SQL_TOOL_DICT, SQLExecuteTool

load_dotenv()


class QuestionGeneratorAgent(DisplayMixin, ContextMixin, ParallelMixin):
    def __init__(
        self, db_path: Path, output_path: Path, model: str = None, config: Dict[str, Any] = None, quiet: bool = False
    ):
        self.db_path = db_path
        self.output_path = output_path
        self.quiet = quiet

        # Load config if not provided
        if config is None:
            config_path = Path(__file__).parent / "config.yaml"
            with open(config_path, "r") as f:
                full_config = yaml.safe_load(f)
                config = full_config.get("agent_question_generator", {})

        self.config = config
        self.model = model or config.get("default_model", "claude-opus-4-5-20251101")

        # Initialize tools
        self.sql_tool = SQLExecuteTool(db_path)
        self.register_tool = RegisterQuestionTool(output_path, db_path)

        # Initialize Claude client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic(api_key=api_key)

        # Load and render system prompt template
        prompt_path = Path(__file__).parent / "prompts" / "agent_system_prompt.j2"
        with open(prompt_path, "r") as f:
            prompt_template = Template(f.read())

        # Load quality rubric
        rubric_path = Path(__file__).parent / "prompts" / "question_quality_rubric.txt"
        with open(rubric_path, "r") as f:
            rubric_content = f.read()

        # Get database overview
        db_overview = self.sql_tool.get_database_overview()

        # Render the template with dynamic data
        self.system_prompt = prompt_template.render(
            schema=db_overview["schema"],
            statistics=db_overview["statistics"],
            sample_ids=db_overview["sample_ids"],
            rubric=rubric_content,
        )

        # Load question type prompt files
        self.type_prompts = {}
        prompts_dir = Path(__file__).parent / "prompts"
        for md_file in prompts_dir.glob("*.md"):
            type_name = md_file.stem  # e.g. "aggregation" from "aggregation.md"
            with open(md_file, "r") as f:
                self.type_prompts[type_name] = f.read()

        # Load question type distribution from config
        full_config_path = Path(__file__).parent / "config.yaml"
        with open(full_config_path, "r") as f:
            full_config = yaml.safe_load(f)
        self.question_type_distribution = full_config.get("generation", {}).get("question_types", {})

        # Optional: Print database stats to show they're loaded
        total_rows = sum(stats["row_count"] for stats in db_overview["statistics"].values())
        print(
            f"{Colors.DIM}Database loaded: {len(db_overview['statistics'])} tables, {total_rows:,} total rows{Colors.ENDC}"
        )
        print(f"{Colors.DIM}Question types loaded: {list(self.type_prompts.keys())}{Colors.ENDC}")

        # Track token usage
        self.total_tokens = {"input": 0, "output": 0}

    # --- Question helpers ---

    def get_existing_questions(self) -> List[Dict[str, str]]:
        """Get list of already generated questions from output file."""
        if not self.output_path.exists():
            return []

        questions = []
        with open(self.output_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        questions.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return questions

    def _format_existing_questions(self, questions: List[Dict[str, str]]) -> str:
        """Format last 10 existing questions for agent context."""
        if not questions:
            return "No questions generated yet. You have complete freedom to be creative!"

        # Only show last 10 questions
        recent_questions = questions[-10:]

        summary = [f"Here are the last {len(recent_questions)} questions (you must generate something different):"]
        summary.append("=" * 80)

        # Include only questions, no answers
        for i, q in enumerate(recent_questions, 1):
            summary.append(f"{i}. {q['question']}")

        summary.append("=" * 80)
        summary.append("Generate a completely different and creative question!")
        return "\n".join(summary)

    def _save_full_trace(self, session_id: str, all_conversations: List[Dict[str, Any]]):
        """Save the complete conversation trace."""
        # Save logs in the same directory as the questions
        trace_path = self.output_path.parent / f"agent_trace_{session_id}.json"

        # Count successful questions
        successful_questions = sum(1 for conv in all_conversations if conv["success"])

        with open(trace_path, "w") as f:
            json.dump(
                {
                    "session_id": session_id,
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "total_questions_target": len(all_conversations),
                    "total_questions_generated": successful_questions,
                    "conversations": all_conversations,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"Saved conversation trace to {trace_path}")

    def _build_type_schedule(self, num_questions: int) -> List[str]:
        """Build a schedule of question types based on the configured distribution."""
        schedule = []
        for type_name, pct in self.question_type_distribution.items():
            count = round(num_questions * pct)
            schedule.extend([type_name] * count)

        # Pad or trim to exact count
        while len(schedule) < num_questions:
            # Add the most common type
            most_common = max(self.question_type_distribution, key=self.question_type_distribution.get)
            schedule.append(most_common)
        schedule = schedule[:num_questions]

        # Shuffle to avoid generating all of one type in a row
        random.shuffle(schedule)
        return schedule

    # --- Core generation ---

    def generate_single_question(
        self,
        question_number: int,
        total: int,
        existing_questions: List[Dict[str, str]],
        max_iterations: int = 100,
        question_type: Optional[str] = None,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Generate a single question with fresh conversation."""

        # Track tokens for this session only
        session_tokens = {"input": 0, "output": 0}

        # Build user message with context about existing questions
        existing_summary = self._format_existing_questions(existing_questions)

        # Build type-specific instruction
        type_instruction = ""
        if question_type and question_type in self.type_prompts:
            type_instruction = (
                f"\n\n**REQUIRED QUESTION TYPE: {question_type}**\n\n"
                f"Follow these type-specific instructions:\n\n"
                f"{self.type_prompts[question_type]}\n\n"
                f"You MUST generate a question of this exact type. Do not deviate.\n"
            )
        elif question_type:
            type_instruction = f"\n\n**REQUIRED QUESTION TYPE: {question_type}**\n"

        messages = [
            {
                "role": "user",
                "content": f"Generate question #{question_number}.\n\n"
                f"{existing_summary}\n\n"
                f"Your task: Create ONE challenging question that requires 3-5 file lookups and is COMPLETELY DIFFERENT from the above."
                f"{type_instruction}\n\n"
                f"Key requirements:\n"
                f"1. Start by exploring different tables/attributes than recent questions\n"
                f"2. Verify exactly ONE correct answer exists (use verification_query)\n"
                f"3. The answer must be a CONCRETE value (name, number, date) — never 'None' or 'does not own'\n"
                f"4. If asking about a pet or job, verify the person has exactly 1 of that type\n"
                f"5. Minimum 3 files required\n\n"
                f"EXPLORATION STRATEGY: Run multiple SQL queries in parallel to explore efficiently!\n"
                f"Example: Check different tables, test various conditions, explore relationships simultaneously.\n\n"
                f"When you find a great question with a unique answer, call register_question ALONE (this ends the session).",
            }
        ]

        conversation_trace = messages.copy()
        question_registered = False
        last_input_tokens = 0  # Track tokens from last API response

        for iteration in range(1, max_iterations + 1):
            self._print_progress(question_number, total, iteration, max_iterations)

            # Trim messages if needed based on last response's input tokens
            messages = self._trim_messages_if_needed(messages, last_input_tokens)

            # Get agent response with error handling
            try:
                response = self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    system=self.system_prompt,
                    max_tokens=self.config.get("max_tokens_per_response", 4096),
                    tool_choice={"type": "any"},
                    tools=[EXECUTE_SQL_TOOL_DICT, REGISTER_QUESTION_TOOL_DICT],
                )

            except Exception as e:
                print(f"\n{Colors.RED}API Error: {str(e)}{Colors.ENDC}")

                # Record error in conversation trace
                error_record = {"role": "error", "message": f"API call failed: {str(e)}", "iteration": iteration}
                conversation_trace.append(error_record)

                # End this question attempt and return failure
                print(f"{Colors.YELLOW}Ending question generation due to error{Colors.ENDC}")
                return False, conversation_trace

            # Print token usage and track last input tokens
            self._print_token_usage(response.usage, session_tokens)
            if response.usage:
                last_input_tokens = response.usage.input_tokens

            # Add assistant message to conversation
            assistant_message = {"role": "assistant", "content": response.content}
            messages.append(assistant_message)
            conversation_trace.append(assistant_message)

            # Process tool calls
            tool_results = []
            tool_calls = [block for block in response.content if block.type == "tool_use"]

            # Safety check: ensure register_question is not called with other tools
            if len(tool_calls) > 1:
                has_register = any(block.name == "register_question" for block in tool_calls)
                if has_register:
                    print(
                        f"{Colors.RED}Error: register_question must be called alone, not in parallel with other tools{Colors.ENDC}"
                    )
                    # Skip this iteration to let the agent try again
                    messages.append(
                        {
                            "role": "user",
                            "content": "Error: register_question must be called by itself, not in parallel with other tools. Please call register_question alone.",
                        }
                    )
                    continue

            # Separate SQL queries from other tools
            try:
                sql_blocks = [(block, block.input["query"]) for block in tool_calls if block.name == "execute_sql"]
                other_blocks = [block for block in tool_calls if block.name != "execute_sql"]
            except (KeyError, AttributeError) as e:
                print(f"{Colors.YELLOW}Warning: Tool call parsing error: {e}. Continuing...{Colors.ENDC}")
                continue

            # Execute SQL queries concurrently if there are multiple
            if len(sql_blocks) > 1:
                if not self.quiet:
                    print(f"\n{Colors.CYAN}Executing {len(sql_blocks)} SQL queries in parallel...{Colors.ENDC}")

                with ThreadPoolExecutor(max_workers=min(len(sql_blocks), 10)) as executor:
                    # Submit all SQL queries
                    future_to_block = {}
                    for block, query in sql_blocks:
                        self._print_tool_call("execute_sql", {"query": query})
                        future = executor.submit(self.sql_tool.execute, query)
                        future_to_block[future] = block

                    # Collect results as they complete
                    for future in future_to_block:
                        block = future_to_block[future]
                        try:
                            result = future.result()
                            self._print_tool_result("execute_sql", result)

                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": self._format_tool_response("execute_sql", result),
                                }
                            )

                            # Record in trace
                            conversation_trace.append(
                                {"role": "tool_call", "tool": "execute_sql", "input": {"query": block.input["query"]}}
                            )
                            conversation_trace.append({"role": "tool_result", "tool": "execute_sql", "result": result})
                        except Exception as e:
                            result = {"success": False, "error": str(e)}
                            self._print_tool_result("execute_sql", result)
                            tool_results.append(
                                {"type": "tool_result", "tool_use_id": block.id, "content": f"Tool error: {str(e)}"}
                            )

            # Process single SQL query or other tools sequentially
            all_sequential_blocks = other_blocks
            if len(sql_blocks) == 1:
                all_sequential_blocks = tool_calls  # Process normally if only one SQL

            for content_block in all_sequential_blocks:
                tool_name = content_block.name
                tool_input = content_block.input

                # Print tool call
                self._print_tool_call(tool_name, tool_input)

                # Record tool call
                tool_call_record = {"role": "tool_call", "tool": tool_name, "input": tool_input}
                conversation_trace.append(tool_call_record)

                # Execute tool with error handling
                try:
                    if tool_name == "execute_sql":
                        result = self.sql_tool.execute(tool_input["query"])
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": self._format_tool_response(tool_name, result),
                            }
                        )
                    elif tool_name == "register_question":
                        result = self.register_tool.register(
                            tool_input["question"],
                            tool_input["sql_queries"],
                            tool_input["answer"],
                            tool_input["answer_reasoning"],
                            tool_input["question_type"],
                            tool_input["required_files"],
                            tool_input.get("verification_query", ""),
                        )
                        if result["success"]:
                            question_registered = True
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": self._format_tool_response(tool_name, result),
                            }
                        )
                except Exception as tool_error:
                    print(f"{Colors.RED}Tool execution error: {tool_error}{Colors.ENDC}")
                    # Create error result
                    result = {"success": False, "error": str(tool_error)}
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool error: {str(tool_error)}",
                        }
                    )

                # Print tool result
                self._print_tool_result(tool_name, result)

                # Record tool result
                tool_result_record = {"role": "tool_result", "tool": tool_name, "result": result}
                conversation_trace.append(tool_result_record)

            # Add tool results to messages if any
            if tool_results:
                tool_message = {"role": "user", "content": tool_results}
                messages.append(tool_message)
                # Note: tool results are already in conversation_trace

            # Check if we successfully registered a question
            if question_registered:
                print(f"\n{Colors.GREEN}Session ended - Question successfully registered!{Colors.ENDC}")
                break

        # If we hit max iterations without registering, force registration
        if (
            not question_registered
            and iteration == max_iterations
            and self.config.get("force_registration_on_max_iterations", True)
        ):
            print(f"\n{Colors.YELLOW}Hit max iterations without registering - forcing registration...{Colors.ENDC}")

            # Add a final user message
            force_message = {
                "role": "user",
                "content": "You've explored enough! You must now register a question immediately. "
                "Pick your best question idea and register it NOW using register_question.",
            }
            messages.append(force_message)
            conversation_trace.append(force_message)

            # Force tool use with register_question
            try:
                response = self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    system=self.system_prompt,
                    max_tokens=self.config.get("max_tokens_per_response", 4096),
                    tool_choice={"type": "tool", "name": "register_question"},
                    tools=[REGISTER_QUESTION_TOOL_DICT],
                )

                # Print token usage
                self._print_token_usage(response.usage, session_tokens)

                # Process the forced registration
                assistant_message = {"role": "assistant", "content": response.content}
                messages.append(assistant_message)
                conversation_trace.append(assistant_message)

                for content_block in response.content:
                    if content_block.type == "tool_use" and content_block.name == "register_question":
                        tool_input = content_block.input

                        # Print tool call
                        self._print_tool_call("register_question", tool_input)

                        # Record tool call
                        tool_call_record = {"role": "tool_call", "tool": "register_question", "input": tool_input}
                        conversation_trace.append(tool_call_record)

                        # Execute registration
                        result = self.register_tool.register(
                            tool_input["question"],
                            tool_input["sql_queries"],
                            tool_input["answer"],
                            tool_input["answer_reasoning"],
                            tool_input["question_type"],
                            tool_input["required_files"],
                            tool_input.get("verification_query", ""),
                        )

                        if result["success"]:
                            question_registered = True

                        # Print tool result
                        self._print_tool_result("register_question", result)

                        # Record tool result
                        tool_result_record = {"role": "tool_result", "tool": "register_question", "result": result}
                        conversation_trace.append(tool_result_record)

                        print(f"\n{Colors.GREEN}Forced registration completed!{Colors.ENDC}")

            except Exception as e:
                print(f"\n{Colors.RED}Error during forced registration: {e}{Colors.ENDC}")
                error_record = {"role": "error", "message": f"Forced registration failed: {str(e)}"}
                conversation_trace.append(error_record)
                # Still return False since we couldn't register

        return question_registered, conversation_trace

    def generate_questions(self, num_questions: int = 10, question_type: Optional[str] = None):
        """Generate questions using the agent - one at a time with fresh context."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_conversations = []

        # Build type schedule
        if question_type:
            # All questions of the same type
            type_schedule = [question_type] * num_questions
        else:
            type_schedule = self._build_type_schedule(num_questions)

        print(f"\n{Colors.HEADER}Starting question generation with {self.model}{Colors.ENDC}")
        print(f"Target: {num_questions} questions")
        print(f"Output directory: {self.output_path.parent}")
        print(f"Questions file: {self.output_path.name}")
        if question_type:
            print(f"Question type: {question_type} (all questions)")
        else:
            type_counts = Counter(type_schedule)
            print(f"Type distribution: {dict(type_counts)}")
        self._print_separator("=")

        for question_num in range(num_questions):
            # Get all existing questions for context
            existing_questions = self.get_existing_questions()
            current_type = type_schedule[question_num]

            print(
                f"\n{Colors.BOLD}Generating question {question_num + 1}/{num_questions} [{current_type}]{Colors.ENDC}"
            )
            print(f"   {Colors.DIM}Existing questions in corpus: {len(existing_questions)}{Colors.ENDC}")
            self._print_separator()

            # Generate one question with fresh context
            max_iterations = self.config.get("max_iterations_per_question", 20)
            success, conversation = self.generate_single_question(
                question_num + 1,
                num_questions,
                existing_questions,
                max_iterations=max_iterations,
                question_type=current_type,
            )

            # Store conversation
            all_conversations.append(
                {"question_number": question_num + 1, "success": success, "conversation": conversation}
            )

            self._print_separator()
            if success:
                print(f"\n{Colors.GREEN}Successfully generated question {question_num + 1}{Colors.ENDC}")
                # Get the newly registered question to show it
                new_questions = self.get_existing_questions()
                if new_questions and len(new_questions) > len(existing_questions):
                    latest = new_questions[-1]
                    print(f"   {Colors.BOLD}Question:{Colors.ENDC} {latest['question']}")
                    print(f"   {Colors.BOLD}Answer:{Colors.ENDC} {latest['answer']}")
            else:
                print(f"\n{Colors.RED}Failed to generate question {question_num + 1}{Colors.ENDC}")

            self._print_separator("=")

        # Save full session trace
        self._save_full_trace(session_id, all_conversations)

        # Final summary
        successful_count = sum(1 for conv in all_conversations if conv["success"])

        print(f"\n{Colors.HEADER}Generation complete!{Colors.ENDC}")
        self._print_separator("=")
        print(f"{Colors.BOLD}Results:{Colors.ENDC}")
        print(
            f"   {Colors.GREEN if successful_count == num_questions else Colors.YELLOW}Successfully generated: {successful_count}/{num_questions} questions{Colors.ENDC}"
        )
        print(f"   Output directory: {self.output_path.parent}")
        print(f"   Questions file: {self.output_path.name}")
        print(f"   Trace file: agent_trace_{session_id}.json")
        print(
            f"   {Colors.DIM}Total tokens used: {self.total_tokens['input']:,} input, {self.total_tokens['output']:,} output{Colors.ENDC}"
        )
        print(
            f"   {Colors.DIM}Estimated cost: ${(self.total_tokens['input'] * 0.003 + self.total_tokens['output'] * 0.015) / 1000:.2f}{Colors.ENDC}"
        )
        self._print_separator("=")


def main():
    parser = argparse.ArgumentParser(description="Generate questions using AI agent")
    parser.add_argument("--num-questions", type=int, default=10, help="Number of questions to generate")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path(__file__).parent / "data" / "letta_file_bench.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data" / "generated_questions",
        help="Output directory for generated questions (will create timestamped subdirectory)",
    )
    parser.add_argument("--model", type=str, default="claude-opus-4-5-20251101", help="Claude model to use")
    parser.add_argument(
        "--question-type",
        type=str,
        default=None,
        choices=[
            "multi_hop_chain",
            "aggregation",
            "set_intersection",
            "negation",
            "comparison_tiebreak",
            "multi_entity_comparison",
            "cross_file_counting",
            "temporal_reasoning",
        ],
        help="Generate all questions of this specific type (default: use distribution from config)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers for question generation (default: 1, sequential)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=True,
        help="Append to the latest existing run instead of creating a new one (default: True)",
    )
    parser.add_argument(
        "--new-run", action="store_true", help="Force creation of a new run directory (overrides --append)"
    )

    args = parser.parse_args()

    # Determine output directory based on append/new-run flags
    if args.new_run or not args.append:
        # Create new timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Creating new run directory: {output_dir}")
    else:
        # Find latest run directory to append to
        if args.output_dir.exists():
            run_dirs = [d for d in args.output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
            if run_dirs:
                output_dir = sorted(run_dirs)[-1]  # Get latest by name (timestamp)
                print(f"Appending to existing run: {output_dir}")
            else:
                # No existing runs, create first one
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = args.output_dir / f"run_{timestamp}"
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"No existing runs found. Creating first run: {output_dir}")
        else:
            # Output directory doesn't exist, create it with first run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = args.output_dir / f"run_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Creating new output directory and first run: {output_dir}")

    # Set output file path
    output_path = output_dir / "agent_generated_questions.jsonl"

    # Verify database exists
    if not args.db_path.exists():
        print(f"Error: Database not found at {args.db_path}")
        print("Please run the JSONL to SQLite conversion first.")
        return

    # Create agent and generate questions
    try:
        agent = QuestionGeneratorAgent(db_path=args.db_path, output_path=output_path, model=args.model)

        print(f"Starting question generation with {args.model}...")
        print(f"Database: {args.db_path}")
        print(f"Output directory: {output_dir}")
        print(f"Questions file: {output_path.name}")
        print(f"Target: {args.num_questions} questions\n")

        if args.parallel > 1:
            agent.generate_questions_parallel(
                args.num_questions,
                num_workers=args.parallel,
                question_type=args.question_type,
            )
        else:
            agent.generate_questions(args.num_questions, question_type=args.question_type)

    except Exception as e:
        print(f"{Colors.RED}Fatal error during initialization: {e}{Colors.ENDC}")
        return


if __name__ == "__main__":
    main()
