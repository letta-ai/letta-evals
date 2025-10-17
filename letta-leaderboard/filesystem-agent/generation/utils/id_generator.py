"""
Utilities for generating unique sequential IDs.
"""
from typing import Dict


class IDGenerator:
    """Generates sequential IDs with prefixes for different entity types."""
    
    def __init__(self):
        self._counters: Dict[str, int] = {}
    
    def reset(self):
        """Reset all counters for a fresh generation run."""
        self._counters.clear()
    
    def generate_id(self, prefix: str) -> str:
        """Generate a unique ID with the given prefix."""
        if prefix not in self._counters:
            self._counters[prefix] = 1
        else:
            self._counters[prefix] += 1
        return f"{prefix}-{self._counters[prefix]:04d}"


# Global instance for backwards compatibility
_global_generator = IDGenerator()


def reset_id_counters():
    """Reset all ID counters - backwards compatibility function."""
    _global_generator.reset()


def generate_unique_id(prefix: str) -> str:
    """Generate unique ID - backwards compatibility function."""
    return _global_generator.generate_id(prefix)