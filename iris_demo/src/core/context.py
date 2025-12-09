"""
Scenario context helpers shared across the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .scenario import Scenario


@dataclass
class ScenarioContext:
    """
    Captures runtime metadata for a scenario run.

    The same Scenario can be executed through different sources (LLM simulation,
    sensor ingestion, replay). The context keeps track of that source and any
    optional metadata that downstream stages should be aware of.
    """

    scenario: Scenario
    source: str = "llm"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_metadata(self, **kwargs: Any) -> "ScenarioContext":
        """Return a shallow copy that merges extra metadata."""
        merged = dict(self.metadata)
        merged.update(kwargs)
        return ScenarioContext(scenario=self.scenario, source=self.source, metadata=merged)
