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
    "google_ai/gemini-3-pro-preview": {
        "prompt_tokens": 2,
        "completion_tokens": 12,
    },
    "openai/gpt-5.1-codex-mini": {
        "prompt_tokens": 0.25,
        "completion_tokens": 2,
    },
    "openai/gpt-5.1-codex": {
        "prompt_tokens": 1.25,
        "completion_tokens": 10,
    },
    "openai/gpt-5.1-2025-11-13": {
        "prompt_tokens": 1.25,
        "completion_tokens": 10,
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


# Build reverse mapping: strip provider prefix and date suffix from MODEL_COSTS keys
# This allows matching "gpt-4.1-mini" to "openai/gpt-4.1-mini-2025-04-14"
def _build_model_name_mapping() -> dict:
    """
    Build a mapping from base model names to full MODEL_COSTS keys.

    Examples:
        "gpt-4.1-mini" -> "openai/gpt-4.1-mini-2025-04-14"
        "claude-sonnet-4-5" -> "anthropic/claude-sonnet-4-5-20250929"
    """
    import re

    mapping = {}
    # Pattern to match date suffixes: -YYYY-MM-DD or -YYYYMMDD
    date_pattern = re.compile(r"-\d{4}-\d{2}-\d{2}$|-\d{8}$")

    for full_key in MODEL_COSTS.keys():
        # Remove provider prefix
        if "/" in full_key:
            model_part = full_key.split("/", 1)[1]
        else:
            model_part = full_key

        # Map the full model part
        mapping[model_part] = full_key

        # Strip date suffix if present
        # e.g., "gpt-4.1-mini-2025-04-14" -> "gpt-4.1-mini"
        base_name = date_pattern.sub("", model_part)
        if base_name != model_part:  # Had a date suffix
            mapping[base_name] = full_key

    return mapping


MODEL_NAME_MAPPING = _build_model_name_mapping()
