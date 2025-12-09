"""
Scenario definitions and helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Scenario:
    """Scenario definition loaded from JSON."""

    name: str
    description: str
    duration_minutes: int = 10

    # Classroom setup
    num_children: int = 6
    num_adults: int = 2

    # Optional: specific child profiles to highlight
    focus_children: Optional[list[Dict[str, Any]]] = None

    # Optional: key moments to ensure are captured
    key_moments: Optional[list[Dict[str, str]]] = None

    @classmethod
    def from_json(cls, path: str) -> "Scenario":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scenario":
        return cls(**data)

    @classmethod
    def from_text(cls, name: str, description: str, **kwargs: Any) -> "Scenario":
        """Create scenario from free-form text description."""
        return cls(name=name, description=description, **kwargs)


def load_scenario(path: str) -> Scenario:
    """Load a scenario from a JSON file."""
    return Scenario.from_json(path)


def load_scenarios_from_directory(directory: str) -> Dict[str, Scenario]:
    """Load all scenarios from a directory."""
    import os

    scenarios: Dict[str, Scenario] = {}

    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(directory, filename)
        try:
            scenario = load_scenario(path)
        except Exception as exc:  # pragma: no cover - best-effort loader
            print(f"Error loading {filename}: {exc}")
            continue

        scenarios[scenario.name] = scenario

    return scenarios
