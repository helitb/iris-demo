"""
Formatting helpers shared across event generation components.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List

from .scenario import Scenario
from .schema import (
    AmbientAudioEvent,
    Layer1Event,
    ObjectEvent,
    PostureEvent,
    ProximityEvent,
    SpeechEvent,
    GazeEvent,
)


def build_layer1_prompt(scenario: Scenario) -> str:
    """Build the user prompt for Layer 1 generation."""
    prompt = f"""Generate Layer 1 sensor events for this classroom scenario:

SCENARIO: {scenario.name}
DURATION: {scenario.duration_minutes} minutes
CHILDREN: {scenario.num_children}
ADULTS: {scenario.num_adults}

DESCRIPTION:
{scenario.description}
"""

    if scenario.focus_children:
        prompt += "\n\nFOCUS CHILDREN (pay special attention to these):\n"
        for child in scenario.focus_children:
            prompt += f"- {child.get('id', 'child_?')}: {child.get('description', '')}\n"

    if scenario.key_moments:
        prompt += "\n\nKEY MOMENTS TO CAPTURE:\n"
        for moment in scenario.key_moments:
            prompt += f"- {moment.get('time', '?')}: {moment.get('description', '')}\n"

    prompt += "\n\nGenerate the event stream now:"

    return prompt


def format_layer1_for_prompt(events: List[Layer1Event], base_time: datetime) -> str:
    """Format Layer 1 events for inclusion in Layer 2 prompt."""
    lines = []

    for event in events:
        offset = (event.timestamp - base_time).total_seconds()

        if isinstance(event, SpeechEvent):
            meta_parts = [f"word_count={event.word_count}", f"type={event.vocal_type.value}"]
            if getattr(event, "complexity", None):
                meta_parts.append(f"complexity={event.complexity.value}")
            if getattr(event, "is_echolalia_candidate", False):
                meta_parts.append(f"echolalia_candidate={event.is_echolalia_candidate}")
            if getattr(event, "echolalia_similarity", None) is not None:
                meta_parts.append(f"echolalia_similarity={event.echolalia_similarity:.2f}")

            transcript_text = event.transcription or "[non-verbal]"
            transcript_serialized = json.dumps(transcript_text, ensure_ascii=False)
            lines.append(
                f"[{offset:.0f}s] {event.event_id} SPEECH: {event.speaker.id} "
                f"{transcript_serialized} "
                f"(target={event.target.value}, metadata: {', '.join(meta_parts)})"
            )

        elif isinstance(event, AmbientAudioEvent):
            lines.append(
                f"[{offset:.0f}s] {event.event_id} AMBIENT: {event.sound_type} "
                f"({event.intensity.value} intensity)"
            )

        elif isinstance(event, ProximityEvent):
            lines.append(
                f"[{offset:.0f}s] {event.event_id} PROXIMITY: {event.actor.id} "
                f"{event.change_type.value} {event.target.id} "
                f"({event.proximity_level.value})"
            )

        elif isinstance(event, GazeEvent):
            target = event.target_actor.id if event.target_actor else event.target_object or "?"
            lines.append(
                f"[{offset:.0f}s] {event.event_id} GAZE: {event.actor.id} "
                f"looking {event.direction.value} at {target}"
                f"{' (mutual)' if event.is_mutual else ''}"
            )

        elif isinstance(event, PostureEvent):
            lines.append(
                f"[{offset:.0f}s] {event.event_id} POSTURE: {event.actor.id} "
                f"{event.posture.value}, {event.movement.value}"
                f"{' (repetitive)' if event.is_repetitive else ''}, "
                f"orientation={event.orientation.value}"
            )

        elif isinstance(event, ObjectEvent):
            shared = f" with {event.shared_with.id}" if event.shared_with else ""
            lines.append(
                f"[{offset:.0f}s] {event.event_id} OBJECT: {event.actor.id} "
                f"{event.action.value} {event.object_type}{shared}"
            )

    return "\n".join(lines)


def build_layer2_prompt(
    layer1_events: List[Layer1Event],
    base_time: datetime,
) -> str:
    """Build the user prompt for Layer 2 generation."""
    l1_summary = format_layer1_for_prompt(layer1_events, base_time)

    user_prompt = f"""Analyze these Layer 1 sensor events and generate Layer 2 interpreted events.

LAYER 1 EVENTS:
{l1_summary}

Generate Layer 2 events (BEHAVIORAL, INTERACTION, CONTEXT) that capture the meaningful patterns:"""

    format_instructions = """
OUTPUT FORMAT (strict):
Emit each Layer 2 event on a SINGLE LINE using the schema tag followed immediately by a JSON object. No extra text, no multi-line JSON. Examples:

BEHAVIORAL|{"timestamp_offset_sec": 0.0, "event_id": "L2_behavioral_001", "actor_id": "child_1", "category": "TASK_ENGAGEMENT", "intensity": "MODERATE", "apparent_emotion": "CALM", "trigger": "TASK_DEMAND", "trigger_description": "...", "description": "...", "regulation_effective": true, "source_event_ids": ["L1_a", "L1_b"], "confidence": 0.8}
INTERACTION|{"timestamp_offset_sec": 12.0, "event_id": "L2_interaction_001", "initiator_id": "child_2", "recipient_id": "child_1", "interaction_type": "COLLABORATION_ATTEMPT", "quality": "UNSUCCESSFUL", "content_type": "REQUEST", "communicative_intent": "SOCIAL_CONNECTION", "description": "...", "reciprocity_level": "...", "adult_facilitated": false, "source_event_ids": ["L1_a", "L1_b"], "confidence": 0.8}
CONTEXT|{"timestamp_offset_sec": 55.0, "event_id": "L2_context_001", "activity_type": "STRUCTURED_ACTIVITY", "activity_description": "...", "primary_zone": "WORK_AREA", "classroom_climate": "CALM", "noise_level": "LOW", "is_transition": false, "adults_present": 1, "adult_attention_focus": "...", "source_event_ids": ["L1_a", "L1_b"], "confidence": 0.8}

Do NOT emit markdown, numbered lists, or prose—only these single-line JSON entries.
"""

    return user_prompt + format_instructions
