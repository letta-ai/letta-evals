"""Context window management for question generation conversations."""

from typing import Any, Dict, List

from display import Colors


class ContextMixin:
    """Mixin providing context window management for QuestionGeneratorAgent.

    Expects the host class to have:
        - self.config: Dict[str, Any]
        - self.model: str
        - self.client: Anthropic client
        - self.quiet: bool
    """

    def _summarize_conversation_with_llm(self, messages_to_summarize: List[Dict[str, Any]]) -> str:
        """Use LLM to create a concise summary of the conversation history."""
        try:
            # Create a summarization prompt
            summary_prompt = """Please summarize the following conversation history into a concise summary.
Focus on:
1. What SQL queries were explored and their key findings
2. What patterns or relationships were discovered
3. What question ideas were considered
4. What remains to be explored

Keep it brief but informative. This summary will help continue the conversation."""

            # Build conversation text for summarization
            conversation_text = []
            for msg in messages_to_summarize:
                if msg["role"] == "assistant":
                    # Extract text content from assistant messages
                    if isinstance(msg["content"], list):
                        for item in msg["content"]:
                            if hasattr(item, "text"):
                                conversation_text.append(f"Assistant: {item.text}")
                    else:
                        conversation_text.append(f"Assistant: {msg['content']}")
                elif msg["role"] == "user":
                    # Handle user messages (including tool results)
                    if isinstance(msg["content"], list):
                        for item in msg["content"]:
                            if isinstance(item, dict) and item.get("type") == "tool_result":
                                conversation_text.append(f"Tool Result: {item.get('content', '')}")
                    else:
                        conversation_text.append(f"User: {msg['content']}")

            # Create messages for summarization - system goes as parameter, not in messages
            max_items = self.config.get("summary_max_conversation_items", 30)
            condensed_messages = [
                {
                    "role": "user",
                    "content": summary_prompt
                    + "\n\nConversation to summarize:\n"
                    + "\n".join(conversation_text[:max_items]),
                }
            ]

            # Call the LLM for summarization
            response = self.client.messages.create(
                model=self.model,
                system="You are a helpful assistant that summarizes conversations.",  # System as parameter
                messages=condensed_messages,
                max_tokens=self.config.get("summary_max_tokens", 500),  # Keep summary concise
                temperature=self.config.get("summary_temperature", 0.3),  # Lower temperature for factual summary
            )

            # Extract the summary text
            summary = (
                response.content[0].text
                if response.content
                else "Previous exploration of database patterns and relationships."
            )

            return f"[Context Summary]\n{summary}\n[End Summary]"

        except Exception as e:
            print(f"{Colors.YELLOW}Warning: Could not generate LLM summary: {e}{Colors.ENDC}")
            # Fallback to basic summary
            return "[Context Summary]\nPrevious exploration included multiple SQL queries and pattern discovery.\n[End Summary]"

    def _trim_messages_if_needed(
        self, messages: List[Dict[str, Any]], last_response_tokens: int = None
    ) -> List[Dict[str, Any]]:
        """Trim older messages if approaching context limit."""
        # Use the input tokens from last response as indicator of current context size
        trim_threshold = self.config.get("trim_threshold", 140000)
        if last_response_tokens is None or last_response_tokens < trim_threshold:
            return messages

        print(
            f"\n{Colors.YELLOW}Approaching token limit ({last_response_tokens:,} tokens in last request) - compressing conversation...{Colors.ENDC}"
        )

        # We need to be careful to keep tool_use/tool_result pairs together
        # Find the last complete exchange (assistant message followed by optional user tool results)
        messages_to_keep = self.config.get("messages_to_keep_on_trim", 6)
        keep_from_index = max(1, len(messages) - messages_to_keep)  # Keep more messages to ensure completeness

        # Ensure we start from an assistant message to maintain pairing
        while keep_from_index < len(messages) - 1 and messages[keep_from_index]["role"] != "assistant":
            keep_from_index += 1

        if keep_from_index > 1:
            # Messages to summarize
            messages_to_summarize = messages[1:keep_from_index]

            # Generate LLM summary
            summary = self._summarize_conversation_with_llm(messages_to_summarize)

            # Build new message list
            new_messages = [messages[0]]  # Keep initial user message

            # Add summary as a user message (safer than assistant)
            summary_message = {"role": "user", "content": summary}
            new_messages.append(summary_message)

            # Keep messages from the cutoff point
            new_messages.extend(messages[keep_from_index:])

            removed = len(messages) - len(new_messages)
            print(f"{Colors.DIM}Compressed {removed} messages into LLM-generated summary{Colors.ENDC}")
            return new_messages

        return messages
