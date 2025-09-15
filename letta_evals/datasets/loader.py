import json
from pathlib import Path
from typing import Iterator, List, Optional

from letta_evals.models import Sample, SampleMetadata


def load_jsonl(
    file_path: Path, max_samples: Optional[int] = None, sample_tags: Optional[List[str]] = None
) -> Iterator[Sample]:
    """Load samples from a JSONL file."""
    with open(file_path, "r") as f:
        count = 0
        for line in f:
            if max_samples and count >= max_samples:
                break

            data = json.loads(line.strip())

            metadata = SampleMetadata()
            if "metadata" in data:
                metadata = SampleMetadata(**data["metadata"])

            if sample_tags:
                if not any(tag in metadata.tags for tag in sample_tags):
                    continue

            sample = Sample(
                input=data["input"],
                ground_truth=data.get("ground_truth"),
                metadata=metadata,
                id=data.get("id"),
                agent_args=data.get("agent_args"),
            )

            yield sample
            count += 1
