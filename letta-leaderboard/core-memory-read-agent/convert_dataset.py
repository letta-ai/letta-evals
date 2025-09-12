#!/usr/bin/env python3
"""
Convert letta_bench_gen_200.jsonl to letta-evals-kit format for core memory read evaluation.
"""
import json
import sys
from pathlib import Path

def convert_dataset():
    """Convert the letta benchmark dataset to letta-evals format."""
    
    # Read the original dataset
    input_file = Path("leaderboard/letta_bench/letta_bench_gen_200.jsonl")
    output_file = Path("core-memory-read-agent/datasets/core_memory_read.jsonl")
    
    converted_samples = []
    
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Extract questions, answers, and facts
            questions = data["question"]
            answers = data["answer"]
            facts = data["facts"]
            names = data["name"]
            
            # Create samples for each question-answer pair
            for i, (question, answer) in enumerate(zip(questions, answers)):
                # Create the core memory context by joining all facts
                facts_context = "\n".join(f"{j}. {fact}" for j, fact in enumerate(facts))
                
                sample = {
                    "input": question,
                    "ground_truth": answer,
                    "metadata": {
                        "tags": [],
                        "extra": {
                            "facts": facts,
                            "names": names,
                            "facts_context": facts_context,
                            "question_index": i
                        }
                    }
                }
                converted_samples.append(sample)
    
    # Write converted samples to output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for sample in converted_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Converted {len(converted_samples)} samples to {output_file}")
    return len(converted_samples)

if __name__ == "__main__":
    convert_dataset()