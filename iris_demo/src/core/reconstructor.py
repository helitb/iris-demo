"""
Scenario reconstruction helpers operating on sanitized Layer 2 events.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal

from .llm import LLMClient, get_config
from .prompting import RECONSTRUCT_SCENARIO_SYSTEM_PROMPT
from .schema import Layer2Event, BehavioralEvent, InteractionEvent, ContextEvent
from .session import ReconstructionArtifact


def _format_l2_events_for_llm(l2_events: List[Layer2Event]) -> str:
    """Format Layer 2 events for LLM consumption."""
    lines = []

    for evt in l2_events:
        if isinstance(evt, BehavioralEvent):
            lines.append(
                f"[{evt.timestamp.strftime('%H:%M:%S')}] BEHAVIORAL: {evt.actor.id} - {evt.description}\n"
                f"  Category: {evt.category.value}, Intensity: {evt.intensity.value}, "
                f"Emotion: {evt.apparent_emotion.value}, Trigger: {evt.trigger.value}"
            )

        elif isinstance(evt, InteractionEvent):
            lines.append(
                f"[{evt.timestamp.strftime('%H:%M:%S')}] INTERACTION: {evt.initiator.id} → {evt.recipient.id} - {evt.description}\n"
                f"  Type: {evt.interaction_type.value}, Quality: {evt.quality.value}"
            )

        elif isinstance(evt, ContextEvent):
            lines.append(
                f"[{evt.timestamp.strftime('%H:%M:%S')}] CONTEXT: {evt.activity_type.value} in {evt.primary_zone.value}\n"
                f"  Climate: {evt.classroom_climate.value}, Noise: {evt.noise_level.value}"
            )

    return "\n".join(lines)


class ScenarioReconstructor:
    """Generates a natural language narrative from sanitized Layer 2 events."""

    def __init__(
        self,
        client: LLMClient | None = None,
        language: Literal["en", "he"] = "en",
    ):
        self.client = client or LLMClient()
        self.config = get_config()
        self.language = language if language in {"en", "he"} else "en"

    def reconstruct(self, l2_events: List[Layer2Event]) -> ReconstructionArtifact:
        l2_formatted = _format_l2_events_for_llm(l2_events)

        language_instruction = (
            "Write the reconstruction in Hebrew."
            if self.language == "he"
            else "Write the reconstruction in English."
        )

        user_prompt = f"""Based on the following Layer 2 (behavioral/social) inferred events, reconstruct a natural language description of what happened in the classroom.

LAYER 2 EVENTS (analyze these to reconstruct the scenario):
{l2_formatted}

Generate a detailed natural language reconstruction of the classroom events that occurred, based on the Layer 2 events.

{language_instruction}"""

        if hasattr(self.client, "stream_prompt"):
            text = self.client.stream_prompt(
                system_prompt=RECONSTRUCT_SCENARIO_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=self.config.report_max_tokens,
            )
        else:
            text = self.client.complete(
                RECONSTRUCT_SCENARIO_SYSTEM_PROMPT,
                user_prompt,
                max_tokens=self.config.report_max_tokens,
            )

        return ReconstructionArtifact(text=text, generated_at=datetime.now())
