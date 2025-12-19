"""
Scenario report utilities 
"""

from typing import Optional, Literal

from .schema import (
    Session,
    SpeechEvent, AmbientAudioEvent, ProximityEvent, GazeEvent,
    PostureEvent, ObjectEvent,
    BehavioralEvent, InteractionEvent, ContextEvent,
)
from .llm import LLMClient, get_config
from .prompting import SOCIAL_STORY_SYSTEM_PROMPT, CLINICAL_REPORT_SYSTEM_PROMPT
from .session import ReconstructionArtifact

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


class ReportGenerator:
    """
    Scenario reports generator.
    """
    
    def __init__(self, 
        client: LLMClient | None = None,
        language: Literal["en", "he"] = "en",
    ):
        self.client = LLMClient() if client is None else client
        self.config = get_config()
        self.language = language if language in {"en", "he"} else "en"
        self.report_type = type
    
    def generate(self, 
        type: Literal["social_story", "slp_clinical"],
        reconstruction: ReconstructionArtifact,
        ) -> ReconstructionArtifact:
        """
        Generate a report from session data.
        
        Args:
            session: Session with events
        
        Returns:
            Generated report text
        """
        if type == "social_story":
            self.SYSTEM_PROMPT = SOCIAL_STORY_SYSTEM_PROMPT
        elif type == "slp_clinical":
            self.SYSTEM_PROMPT = CLINICAL_REPORT_SYSTEM_PROMPT
        else:
            raise ValueError(f"Unknown report type: {type}")
        
        language_instruction = (
            "Write the report in Hebrew."
            if self.language == "he"
            else "Write the report in English."
        )

        
        user_prompt = f"""{reconstruction.text}

{language_instruction}
Generate the report now:"""
        
        text = self.client.complete(
            self.SYSTEM_PROMPT,
            user_prompt,
            max_tokens=self.config.report_max_tokens,
        )
        return ReconstructionArtifact(text=text.strip())
