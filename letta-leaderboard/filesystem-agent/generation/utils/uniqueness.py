"""
Utilities for ensuring uniqueness of generated data fields.
"""
from typing import Set, Dict, Callable


class UniquenessTracker:
    """Tracks unique values across different categories to prevent duplicates."""
    
    def __init__(self):
        self._used_values: Dict[str, Set[str]] = {
            'person_names': set(),
            'company_names': set(),
            'usernames': set(),
            'license_plates': set(),
            'credit_card_numbers': set(),
            'ssns': set(),
            'emails': set(),
            'phone_numbers': set(),
            'policy_numbers': set(),
            'account_numbers': set(),
        }
    
    def reset(self):
        """Reset all tracking sets for a fresh generation run."""
        for value_set in self._used_values.values():
            value_set.clear()
    
    def ensure_unique(self, generator: Callable[[], str], category: str, max_attempts: int = 100) -> str:
        """Generate a unique value using the provided generator function."""
        if category not in self._used_values:
            raise ValueError(f"Unknown category: {category}")
            
        for _ in range(max_attempts):
            value = generator()
            if value not in self._used_values[category]:
                self._used_values[category].add(value)
                return value
        
        raise ValueError(f"Could not generate unique {category} after {max_attempts} attempts")


# Global instance for backwards compatibility
_global_tracker = UniquenessTracker()


def reset_uniqueness_tracking():
    """Reset all uniqueness tracking - backwards compatibility function."""
    _global_tracker.reset()


def ensure_unique_value(generator: Callable[[], str], category: str, max_attempts: int = 100) -> str:
    """Ensure unique value - backwards compatibility function."""
    return _global_tracker.ensure_unique(generator, category, max_attempts)