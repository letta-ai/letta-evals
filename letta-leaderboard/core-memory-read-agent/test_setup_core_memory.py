"""
Unit tests for the setup_core_memory function using real evaluation data.
"""
import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from setup_agent import setup_core_memory
from letta_evals.models import Sample, SampleMetadata
from letta_evals.datasets.loader import load_jsonl


class TestSetupCoreMemory:
    """Test cases for setup_core_memory function using real evaluation data."""

    @classmethod
    def setup_class(cls):
        """Load real samples from the evaluation dataset."""
        dataset_path = Path(__file__).parent / "datasets" / "core_memory_read.jsonl"
        cls.real_samples = list(load_jsonl(dataset_path, max_samples=5))  # Load first 5 samples for testing

    def create_mock_sample(self, facts: List[str], ground_truth: str = "test answer") -> Sample:
        """Create a mock Sample object with the given facts."""
        return Sample(
            input="test question",
            ground_truth=ground_truth,
            metadata=SampleMetadata(
                tags=[],
                extra={"facts": facts}
            )
        )

    def create_mock_memory_block(self, block_id: str = "block1", label: str = "persona", value: str = ""):
        """Create a mock memory block."""
        mock_block = MagicMock()
        mock_block.id = block_id
        mock_block.label = label
        mock_block.value = value
        return mock_block

    def create_mock_agent_state(self, memory_blocks: List = None):
        """Create a mock agent state with memory blocks."""
        mock_agent = MagicMock()
        mock_memory = MagicMock()
        mock_memory.blocks = memory_blocks or []
        mock_agent.memory = mock_memory
        return mock_agent

    @pytest.mark.asyncio
    async def test_setup_core_memory_success(self):
        """Test successful memory setup with facts."""
        # Arrange
        facts = ["fact 1", "fact 2", "fact 3"]
        sample = self.create_mock_sample(facts)

        mock_client = AsyncMock()
        mock_agent_state = self.create_mock_agent_state([
            self.create_mock_memory_block("block1", "persona")
        ])
        mock_client.agents.retrieve.return_value = mock_agent_state
        mock_client.agents.blocks.modify = AsyncMock()

        # Act
        await setup_core_memory(mock_client, "test_agent_id", sample)

        # Assert
        mock_client.agents.retrieve.assert_called_once_with("test_agent_id")
        mock_client.agents.blocks.modify.assert_called_once_with(
            agent_id="test_agent_id",
            block_label="persona",
            label="Supporting Facts",
            value="1. fact 1\n2. fact 2\n3. fact 3"
        )

    @pytest.mark.asyncio
    async def test_setup_core_memory_no_facts(self):
        """Test that function raises ValueError when no facts are provided."""
        # Arrange
        sample = self.create_mock_sample([])  # Empty facts list
        mock_client = AsyncMock()

        # Act & Assert
        with pytest.raises(ValueError, match="No facts available for sample"):
            await setup_core_memory(mock_client, "test_agent_id", sample)

    @pytest.mark.asyncio
    async def test_setup_core_memory_no_metadata(self):
        """Test that function raises ValueError when sample has no metadata."""
        # Arrange
        sample = Sample(
            input="test question",
            ground_truth="test answer",
            metadata=None
        )
        mock_client = AsyncMock()

        # Act & Assert
        with pytest.raises(ValueError, match="No facts available for sample"):
            await setup_core_memory(mock_client, "test_agent_id", sample)

    @pytest.mark.asyncio
    async def test_setup_core_memory_no_extra_field(self):
        """Test that function raises ValueError when metadata has no extra field."""
        # Arrange
        sample = Sample(
            input="test question",
            ground_truth="test answer",
            metadata=SampleMetadata(tags=[], extra=None)
        )
        mock_client = AsyncMock()

        # Act & Assert
        with pytest.raises(ValueError, match="No facts available for sample"):
            await setup_core_memory(mock_client, "test_agent_id", sample)

    @pytest.mark.asyncio
    async def test_setup_core_memory_no_memory_blocks(self):
        """Test that function returns early when agent has no memory blocks."""
        # Arrange
        facts = ["fact 1", "fact 2"]
        sample = self.create_mock_sample(facts)

        mock_client = AsyncMock()
        mock_agent_state = self.create_mock_agent_state([])  # No memory blocks
        mock_client.agents.retrieve.return_value = mock_agent_state

        # Act
        await setup_core_memory(mock_client, "test_agent_id", sample)

        # Assert
        mock_client.agents.retrieve.assert_called_once_with("test_agent_id")
        mock_client.agents.blocks.modify.assert_not_called()

    @pytest.mark.asyncio
    async def test_setup_core_memory_no_memory_object(self):
        """Test that function returns early when agent has no memory object."""
        # Arrange
        facts = ["fact 1", "fact 2"]
        sample = self.create_mock_sample(facts)

        mock_client = AsyncMock()
        mock_agent_state = MagicMock()
        mock_agent_state.memory = None
        mock_client.agents.retrieve.return_value = mock_agent_state

        # Act
        await setup_core_memory(mock_client, "test_agent_id", sample)

        # Assert
        mock_client.agents.retrieve.assert_called_once_with("test_agent_id")
        mock_client.agents.blocks.modify.assert_not_called()

    @pytest.mark.asyncio
    async def test_setup_core_memory_api_error(self):
        """Test that function re-raises API errors."""
        # Arrange
        facts = ["fact 1", "fact 2"]
        sample = self.create_mock_sample(facts)

        mock_client = AsyncMock()
        mock_agent_state = self.create_mock_agent_state([
            self.create_mock_memory_block("block1", "persona")
        ])
        mock_client.agents.retrieve.return_value = mock_agent_state
        mock_client.agents.blocks.modify.side_effect = Exception("API Error")

        # Act & Assert
        with pytest.raises(Exception, match="API Error"):
            await setup_core_memory(mock_client, "test_agent_id", sample)

    @pytest.mark.asyncio
    async def test_setup_core_memory_facts_formatting(self):
        """Test that facts are properly formatted as numbered list."""
        # Arrange
        facts = ["First fact", "Second fact with more details", "Third fact"]
        sample = self.create_mock_sample(facts)

        mock_client = AsyncMock()
        mock_agent_state = self.create_mock_agent_state([
            self.create_mock_memory_block("block1", "persona")
        ])
        mock_client.agents.retrieve.return_value = mock_agent_state

        # Act
        await setup_core_memory(mock_client, "test_agent_id", sample)

        # Assert
        expected_facts = "1. First fact\n2. Second fact with more details\n3. Third fact"
        mock_client.agents.blocks.modify.assert_called_once_with(
            agent_id="test_agent_id",
            block_label="persona",
            label="Supporting Facts",
            value=expected_facts
        )

    @pytest.mark.asyncio
    async def test_setup_core_memory_uses_first_block(self):
        """Test that function uses the first available memory block."""
        # Arrange
        facts = ["fact 1"]
        sample = self.create_mock_sample(facts)

        mock_client = AsyncMock()
        blocks = [
            self.create_mock_memory_block("block1", "persona"),
            self.create_mock_memory_block("block2", "human"),
            self.create_mock_memory_block("block3", "archival")
        ]
        mock_agent_state = self.create_mock_agent_state(blocks)
        mock_client.agents.retrieve.return_value = mock_agent_state

        # Act
        await setup_core_memory(mock_client, "test_agent_id", sample)

        # Assert
        mock_client.agents.blocks.modify.assert_called_once_with(
            agent_id="test_agent_id",
            block_label="persona",  # Should use first block's label
            label="Supporting Facts",
            value="1. fact 1"
        )

    @pytest.mark.asyncio
    async def test_setup_core_memory_verification_success(self):
        """Test that function performs verification after updating memory."""
        # Arrange
        facts = ["fact 1"]
        sample = self.create_mock_sample(facts)

        mock_client = AsyncMock()
        original_block = self.create_mock_memory_block("block1", "persona")
        updated_block = self.create_mock_memory_block("block1", "Supporting Facts", "1. fact 1")

        mock_agent_state = self.create_mock_agent_state([original_block])
        updated_agent_state = self.create_mock_agent_state([updated_block])

        mock_client.agents.retrieve.side_effect = [mock_agent_state, updated_agent_state]

        # Act
        await setup_core_memory(mock_client, "test_agent_id", sample)

        # Assert
        assert mock_client.agents.retrieve.call_count == 2
        mock_client.agents.blocks.modify.assert_called_once()