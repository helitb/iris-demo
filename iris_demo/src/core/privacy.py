"""
Privacy helpers for sanitizing events and sessions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .schema import (
    Layer2Event,
    BehavioralEvent,
    InteractionEvent,
    ContextEvent,
    Session,
)


@dataclass
class PrivacyValidationReport:
    """Report on L2 events privacy validation for reconstruction pipeline."""

    timestamp: datetime
    total_l2_events: int
    events_validated: int
    events_passed: int
    events_failed: int
    failed_event_ids: List[str] = field(default_factory=list)
    failed_event_details: Dict[str, str] = field(default_factory=dict)

    @property
    def validation_success_rate(self) -> float:
        if self.events_validated == 0:
            return 1.0
        return self.events_passed / self.events_validated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_l2_events": self.total_l2_events,
            "events_validated": self.events_validated,
            "events_passed": self.events_passed,
            "events_failed": self.events_failed,
            "validation_success_rate": f"{self.validation_success_rate * 100:.1f}%",
            "failed_event_ids": self.failed_event_ids,
            "failed_event_details": self.failed_event_details,
        }

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "PRIVACY VALIDATION REPORT",
            "=" * 60,
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Total L2 events: {self.total_l2_events}",
            f"Events validated: {self.events_validated}",
            f"Events passed: {self.events_passed}",
            f"Events failed: {self.events_failed}",
            f"Success rate: {self.validation_success_rate * 100:.1f}%",
        ]

        if self.events_failed > 0:
            lines.append("\nFailed events:")
            for event_id in self.failed_event_ids:
                detail = self.failed_event_details.get(event_id, "Unknown issue")
                lines.append(f"  - {event_id}: {detail}")
        else:
            lines.append("\n✅ All events passed privacy validation!")

        return "\n".join(lines)


def validate_layer2_no_pii(event: Layer2Event) -> bool:
    """Return True only if a Layer 2 event contains no quoted speech/PII."""

    pii_indicators = ['"', "'", "said", "stated", "reported", "uttered"]

    def has_pii_text(text: Optional[str]) -> bool:
        if not text:
            return False
        text_lower = text.lower()
        if '"' in text or "'" in text:
            return True
        return any(indicator in text_lower for indicator in ["said", "stated", "uttered"])

    if isinstance(event, BehavioralEvent):
        return not (
            has_pii_text(event.description)
            or has_pii_text(event.trigger_description)
        )
    if isinstance(event, InteractionEvent):
        return not has_pii_text(event.description)
    if isinstance(event, ContextEvent):
        return not has_pii_text(event.activity_description)
    return True


def validate_l2_events(l2_events: List[Layer2Event]) -> PrivacyValidationReport:
    """Validate Layer 2 events for privacy compliance."""

    report = PrivacyValidationReport(
        timestamp=datetime.now(),
        total_l2_events=len(l2_events),
        events_validated=len(l2_events),
        events_passed=0,
        events_failed=0,
    )

    for event in l2_events:
        if validate_layer2_no_pii(event):
            report.events_passed += 1
        else:
            report.events_failed += 1
            report.failed_event_ids.append(event.event_id)
            report.failed_event_details[event.event_id] = (
                f"{type(event).__name__}: {getattr(event, 'description', 'N/A')}"
            )

    return report



def anonymized_session(session: Session) -> Session:
    """Return copy of a session with Layer 1 data removed and L2 validated."""

    anon_session = deepcopy(session)

    anon_session.speech_events.clear()
    anon_session.ambient_audio_events.clear()
    anon_session.proximity_events.clear()
    anon_session.gaze_events.clear()
    anon_session.posture_events.clear()
    anon_session.object_events.clear()

    for event in anon_session.layer2_events:
        if not validate_layer2_no_pii(event):
            print(f"Warning: Layer 2 event {event.event_id} may contain PII: {event}")

    return anon_session


def layer1_only_session(session: Session) -> Session:
    """Return copy of a session containing only Layer 1 events."""

    l1_session = deepcopy(session)
    l1_session.behavioral_events.clear()
    l1_session.interaction_events.clear()
    l1_session.context_events.clear()
    return l1_session
