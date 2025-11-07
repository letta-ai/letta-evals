"""Constants used across the letta_evals."""

# CLI constants
MAX_SAMPLES_DISPLAY = 50

# Model pricing configuration (costs per million tokens)
MODEL_COSTS = {
    "anthropic/claude-opus-4-1-20250805": {
        "prompt_tokens": 15,
        "completion_tokens": 75,
    },
    "anthropic/claude-sonnet-4-5-20250929": {
        "prompt_tokens": 3,
        "completion_tokens": 15,
    },
    "anthropic/claude-haiku-4-5-20251001": {
        "prompt_tokens": 1,
        "completion_tokens": 5,
    },
    "openai/gpt-5-2025-08-07": {
        "prompt_tokens": 1.25,
        "completion_tokens": 10,
    },
    "openai/gpt-5-mini-2025-08-07": {
        "prompt_tokens": 0.25,
        "completion_tokens": 2,
    },
    "openai/gpt-5-nano-2025-08-07": {
        "prompt_tokens": 0.05,
        "completion_tokens": 0.4,
    },
    "openai/gpt-4.1-2025-04-14": {
        "prompt_tokens": 2,
        "completion_tokens": 8,
    },
    "openai/gpt-4.1-mini-2025-04-14": {
        "prompt_tokens": 0.4,
        "completion_tokens": 1.6,
    },
    "openai/gpt-4.1-nano-2025-04-14": {
        "prompt_tokens": 0.10,
        "completion_tokens": 0.4,
    },
    "deepseek/deepseek-chat-v3.1": {
        "prompt_tokens": 0.27,
        "completion_tokens": 1,
    },
    "moonshotai/kimi-k2-0905": {
        "prompt_tokens": 0.39,
        "completion_tokens": 1.9,
    },
    "z-ai/glm-4.6": {
        "prompt_tokens": 0.5,
        "completion_tokens": 1.75,
    },
    "openai/gpt-oss-120b": {
        "prompt_tokens": 0.15,
        "completion_tokens": 0.6,
    },
    "openai/gpt-oss-20b": {
        "prompt_tokens": 0.05,
        "completion_tokens": 0.2,
    },
}
