"""
Base report utilities and abstract class.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..schema import (
    Session,
    SpeechEvent, AmbientAudioEvent, ProximityEvent, GazeEvent,
    PostureEvent, ObjectEvent,
    BehavioralEvent, InteractionEvent, ContextEvent,
)
from ..llm import LLMClient, get_config


def format_events_for_report(session: Session, max_layer1_events: int = 50) -> str:
    """
    Format session events as text for report generation.
    
    Args:
        session: Session with events
        max_layer1_events: Limit Layer 1 events to avoid token overflow
    
    Returns:
        Formatted string of events
    """
    lines = []
    base_time = session.metadata.start_time
    
    # Layer 1 summary
    lines.append("=== LAYER 1: Raw Observations ===\n")
    
    for event in session.layer1_events[:max_layer1_events]:
        offset = (event.timestamp - base_time).total_seconds()
        
        if isinstance(event, SpeechEvent):
            lines.append(
                f"[{offset:.0f}s] SPEECH: {event.speaker.id} said "
                f"'{event.transcription or '[vocalization]'}' "
                f"(target={event.target.value}, complexity={event.complexity.value})"
            )
        elif isinstance(event, AmbientAudioEvent):
            lines.append(
                f"[{offset:.0f}s] SOUND: {event.sound_type} ({event.intensity.value})"
            )
        elif isinstance(event, ProximityEvent):
            lines.append(
                f"[{offset:.0f}s] PROXIMITY: {event.actor.id} "
                f"{event.change_type.value} {event.target.id}"
            )
        elif isinstance(event, GazeEvent):
            target = event.target_actor.id if event.target_actor else event.target_object or "?"
            lines.append(
                f"[{offset:.0f}s] GAZE: {event.actor.id} {event.direction.value} at {target}"
                f"{' (mutual)' if event.is_mutual else ''}"
            )
        elif isinstance(event, PostureEvent):
            lines.append(
                f"[{offset:.0f}s] POSTURE: {event.actor.id} {event.posture.value}, "
                f"{event.movement.value}{' (repetitive)' if event.is_repetitive else ''}"
            )
        elif isinstance(event, ObjectEvent):
            lines.append(
                f"[{offset:.0f}s] OBJECT: {event.actor.id} {event.action.value} {event.object_type}"
            )
    
    # Layer 2 summary
    lines.append("\n=== LAYER 2: Interpreted Events ===\n")
    
    for event in session.layer2_events:
        offset = (event.timestamp - base_time).total_seconds()
        
        if isinstance(event, BehavioralEvent):
            lines.append(
                f"[{offset:.0f}s] BEHAVIOR: {event.actor.id} - {event.category.value}: "
                f"{event.description} (emotion={event.apparent_emotion.value}, "
                f"trigger={event.trigger.value})"
            )
        elif isinstance(event, InteractionEvent):
            lines.append(
                f"[{offset:.0f}s] INTERACTION: {event.initiator.id}→{event.recipient.id} "
                f"{event.interaction_type.value}: {event.description} ({event.quality.value})"
            )
        elif isinstance(event, ContextEvent):
            lines.append(
                f"[{offset:.0f}s] CONTEXT: {event.activity_type.value} @ "
                f"{event.primary_zone.value}, climate={event.classroom_climate.value}"
            )
    
    return "\n".join(lines)


def format_session_header(session: Session) -> str:
    """Format session metadata for report prompts."""
    duration = "N/A"
    if session.metadata.end_time and session.metadata.start_time:
        duration = str(session.metadata.end_time - session.metadata.start_time)
    
    return f"""Session: {session.metadata.scenario_name}
Description: {session.metadata.scenario_description}
Duration: {duration}
Children: {session.metadata.num_children}, Adults: {session.metadata.num_adults}"""


class ReportGenerator(ABC):
    """
    Abstract base class for report generators.
    
    Subclasses must implement:
        - SYSTEM_PROMPT: The system prompt for the LLM
        - report_type: The report type identifier
    """
    
    SYSTEM_PROMPT: str = ""
    report_type: str = ""
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize report generator.
        
        Args:
            model: Optional model override. If None, uses config default.
        """
        self.client = LLMClient(model=model)
        self.config = get_config()
    
    def generate(self, session: Session) -> str:
        """
        Generate a report from session data.
        
        Args:
            session: Session with events
        
        Returns:
            Generated report text
        """
        events_summary = format_events_for_report(session)
        header = format_session_header(session)
        
        user_prompt = f"""{header}

EVENT LOG:
{events_summary}

Generate the report now:"""
        
        return self.client.complete(
            self.SYSTEM_PROMPT,
            user_prompt,
            max_tokens=self.config.report_max_tokens,
        )
