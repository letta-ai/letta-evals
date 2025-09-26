import datetime
from pathlib import Path
from typing import Optional

from letta_client import AsyncLetta, LlmConfig, MessageCreate

from letta_evals.models import Sample, TargetResult
from letta_evals.targets.base import Target
from letta_evals.types import ProgressCallback
from letta_evals.utils import load_object


class AgentTarget(Target):
    """Letta agent target for evaluation."""

    def __init__(
        self,
        client: AsyncLetta,
        agent_id: str = None,
        agent_file: Path = None,
        agent_script: str = None,
        base_dir: Path = None,
        llm_config: Optional[LlmConfig] = None,
    ):
        self.client = client
        self.agent_id = agent_id
        self.agent_file = agent_file
        self.agent_script = agent_script
        self.base_dir = base_dir or Path.cwd()
        self.llm_config = llm_config

    async def run(self, sample: Sample, progress_callback: Optional[ProgressCallback] = None) -> TargetResult:
        """Run the agent on a sample."""
        agent_id = self.agent_id

        if self.agent_file:
            with open(self.agent_file, "rb") as f:
                resp = await self.client.agents.import_file(
                    file=f, append_copy_suffix=False, override_existing_tools=False
                )
                if len(resp.agent_ids) > 1:
                    raise RuntimeError(
                        f"Expected single agent from .af file, got {len(resp.agent_ids)} agents. We don't support multi-agent evals yet."
                    )

                agent_id = resp.agent_ids[0]

        elif self.agent_script:
            agent_factory_func = load_object(self.agent_script, self.base_dir)
            agent_id = await agent_factory_func(self.client, sample)

        if self.llm_config and agent_id:
            await self.client.agents.modify(agent_id=agent_id, llm_config=self.llm_config)

        agent = await self.client.agents.retrieve(agent_id=agent_id, include_relationships=[])
        model_name = self.llm_config.model if self.llm_config else agent.llm_config.model

        # notify progress callback with model name
        if progress_callback and (self.agent_file or self.agent_script):
            await progress_callback.agent_loading(sample.id, model_name=model_name)

        trajectory = []

        # Check if there's a contradicting fact to send before the questions
        contradicting_fact = None
        if sample.agent_args and 'extra' in sample.agent_args and sample.agent_args['extra']:
            contradicting_fact = sample.agent_args['extra'].get('contradicting_fact', None)

        inputs = sample.input if isinstance(sample.input, list) else [sample.input]
        total_messages = len(inputs)
        
        # Send contradicting fact first if available
        if contradicting_fact:
            stream = self.client.agents.messages.create_stream(
                agent_id=agent_id,
                messages=[MessageCreate(role="user", content=f"Please update your knowledge with this new information: {contradicting_fact}")],
                stream_tokens=True
            )
            
            # Process the stream to ensure the message is sent
            async for chunk in stream:
                # Process each chunk as needed - we just need to consume the stream
                pass

        for i, input_msg in enumerate(inputs):
            if progress_callback:
                await progress_callback.message_sending(sample.id, i + 1, total_messages, model_name=model_name)

            stream = self.client.agents.messages.create_stream(
                agent_id=agent_id,
                messages=[MessageCreate(role="user", content=input_msg)],
                stream_tokens=True
            )

            messages = []
            current_message_chunks = []
            current_message_id = None
            
            async for chunk in stream:
                print(chunk)
                # skip non-message types like stop_reason and usage_statistics
                if hasattr(chunk, "message_type"):
                    if chunk.message_type in ["stop_reason", "usage_statistics", "ping"]:
                        continue
                    
                    # Check if this is a new message (different ID)
                    chunk_id = getattr(chunk, 'id', None)
                    if current_message_id != chunk_id:
                        # Process previous message chunks if any exist
                        if current_message_chunks:
                            combined_message = self._combine_message_chunks(current_message_chunks)
                            messages.append(combined_message)
                        
                        # Start new message
                        current_message_chunks = [chunk]
                        current_message_id = chunk_id
                    else:
                        # Same message, add to chunks
                        current_message_chunks.append(chunk)
            
            # Process the last message chunks
            if current_message_chunks:
                combined_message = self._combine_message_chunks(current_message_chunks)
                messages.append(combined_message)

            trajectory.append(messages)
            


        return TargetResult(trajectory=trajectory, agent_id=agent_id, model_name=model_name)

    def _combine_message_chunks(self, chunks):
        """Combine multiple message chunks with the same ID into a single message."""
        if not chunks:
            return None
        
        # Use the first chunk as the base
        base_chunk = chunks[0]
        
        # Accumulate content from all chunks
        combined_content = ""
        for chunk in chunks:
            if hasattr(chunk, 'content') and chunk.content:
                combined_content += str(chunk.content)
        
        # Create a new message object with combined content
        # Since the objects are frozen, we need to create a completely new instance
        try:
            # Use model_dump to get a clean dictionary representation
            chunk_dict = base_chunk.model_dump()
            chunk_dict['content'] = combined_content
            
            # Create a new instance using model_validate
            return type(base_chunk).model_validate(chunk_dict)
        except Exception as e:
            # If model_dump fails, try to manually extract only the essential fields
            try:
                # Extract only the basic fields we need, avoiding FieldInfo and other complex objects
                essential_fields = {}
                for field_name in ['id', 'message_type', 'content', 'date', 'name', 'otid', 
                                 'sender_id', 'step_id', 'is_err', 'seq_id', 'run_id']:
                    if hasattr(base_chunk, field_name):
                        value = getattr(base_chunk, field_name)
                        # Only include simple, serializable values
                        if value is None or isinstance(value, (str, int, float, bool, datetime.datetime)):
                            essential_fields[field_name] = value
                
                essential_fields['content'] = combined_content
                
                # Create a new instance
                return type(base_chunk).model_validate(essential_fields)
            except Exception:
                # Last resort: return the base chunk as-is (this shouldn't happen in normal operation)
                return base_chunk
