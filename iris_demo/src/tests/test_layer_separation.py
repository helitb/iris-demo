"""
Tests for Layer 1 (raw sensor) / Layer 2 (inferred semantic) separation.

Validates:
1. Layer 1 events contain transcriptions (for internal use)
2. Layer 2 prompt never includes raw transcriptions
3. Layer 2 events have no PII or quoted speech
4. Anonymized sessions properly strip Layer 1 data
"""

import pytest
from datetime import datetime, timedelta

from src.core.schema import (
    Actor, ActorRole, Location, ClassroomZone, Intensity,
    SpeechEvent, AmbientAudioEvent, ProximityEvent, GazeEvent,
    PostureEvent, ObjectEvent,
    BehavioralEvent, InteractionEvent, ContextEvent,
    Session, SessionMetadata,
    SpeechTarget, VocalType, VerbalComplexity,
    VocalType, BehaviorCategory, EmotionalState, TriggerType,
    InteractionType, InteractionQuality,
    ActivityType, ClassroomClimate,
)
from src.core.privacy import (
    validate_layer2_no_pii,
    anonymized_session,
    layer1_only_session,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_actors():
    """Create sample anonymous actors."""
    return {
        "child_1": Actor(id="child_1", role=ActorRole.CHILD),
        "child_2": Actor(id="child_2", role=ActorRole.CHILD),
        "adult_1": Actor(id="adult_1", role=ActorRole.ADULT),
    }


@pytest.fixture
def sample_session(sample_actors):
    """Create a sample session with Layer 1 and Layer 2 events."""
    base_time = datetime.now()
    
    # Layer 1: Speech event with transcription
    speech_event = SpeechEvent(
        event_id="L1_speech_001",
        timestamp=base_time,
        speaker=sample_actors["child_1"],
        transcription="אני רוצה את הכדור",  # "I want the ball" in Hebrew
        word_count=4,
        complexity=VerbalComplexity.PHRASE,
        vocal_type=VocalType.SPEECH,
        target=SpeechTarget.PEER,
        duration_ms=800,
    )
    
    # Layer 1: Proximity event
    proximity_event = ProximityEvent(
        event_id="L1_prox_001",
        timestamp=base_time + timedelta(seconds=1),
        actor=sample_actors["child_1"],
        target=sample_actors["child_2"],
        change_type="approach",
        proximity_level="personal",
        actor_location=Location(zone=ClassroomZone.PLAY_AREA),
    )
    
    # Layer 1: Gaze event
    gaze_event = GazeEvent(
        event_id="L1_gaze_001",
        timestamp=base_time + timedelta(seconds=2),
        actor=sample_actors["child_1"],
        direction="at_person",
        target_actor=sample_actors["child_2"],
        duration_ms=500,
        is_mutual=True,
    )
    
    # Layer 2: Behavioral event (no transcription)
    behavioral_event = BehavioralEvent(
        event_id="L2_behavior_001",
        timestamp=base_time + timedelta(seconds=1),
        actor=sample_actors["child_1"],
        category=BehaviorCategory.SOCIAL_APPROACH,
        description="approached peer",  # Abstract, no speech quoted
        intensity=Intensity.MODERATE,
        apparent_emotion=EmotionalState.HAPPY,
        trigger=TriggerType.PEER_ACTION,
        trigger_description="peer was playing with toy",  # No quotes, no speech
        source_event_ids=["L1_speech_001", "L1_prox_001"],
        confidence=0.85,
    )
    
    # Layer 2: Interaction event
    interaction_event = InteractionEvent(
        event_id="L2_interaction_001",
        timestamp=base_time + timedelta(seconds=1),
        initiator=sample_actors["child_1"],
        recipient=sample_actors["child_2"],
        interaction_type=InteractionType.INITIATION,
        description="initiated peer interaction with eye contact",  # No transcription
        quality=InteractionQuality.SUCCESSFUL,
        reciprocity_level="moderate",
        adult_facilitated=False,
        source_event_ids=["L1_speech_001", "L1_prox_001", "L1_gaze_001"],
        confidence=0.88,
    )
    
    # Layer 2: Context event
    context_event = ContextEvent(
        event_id="L2_context_001",
        timestamp=base_time,
        activity_type=ActivityType.FREE_PLAY,
        activity_description="children playing with toys",  # Abstract
        primary_zone=ClassroomZone.PLAY_AREA,
        classroom_climate=ClassroomClimate.ENERGETIC,
        noise_level=Intensity.MODERATE,
        is_transition=False,
        adults_present=1,
        source_event_ids=["L1_speech_001", "L1_prox_001"],
        confidence=0.9,
    )
    
    # Create session
    session = Session(
        metadata=SessionMetadata(
            session_id="test_session_001",
            start_time=base_time,
            end_time=base_time + timedelta(minutes=5),
            scenario_name="Test Scenario",
            scenario_description="A test scenario",
            num_children=2,
            num_adults=1,
        ),
        actors=list(sample_actors.values()),
    )
    
    # Add events
    session.speech_events.append(speech_event)
    session.proximity_events.append(proximity_event)
    session.gaze_events.append(gaze_event)
    session.behavioral_events.append(behavioral_event)
    session.interaction_events.append(interaction_event)
    session.context_events.append(context_event)
    
    return session


# =============================================================================
# TESTS: Layer 1 Contains PII (Transcriptions)
# =============================================================================

def test_layer1_speech_has_transcription(sample_session):
    """Layer 1 speech events should contain verbatim transcriptions."""
    speech_events = sample_session.speech_events
    assert len(speech_events) > 0
    
    speech = speech_events[0]
    assert speech.transcription is not None
    assert speech.transcription == "אני רוצה את הכדור"
    assert len(speech.transcription) > 0


def test_layer1_only_session_has_transcriptions(sample_session):
    """A Layer 1-only session should preserve transcriptions."""
    l1_session = layer1_only_session(sample_session)
    
    # Layer 1 events present
    assert len(l1_session.layer1_events) > 0
    # Layer 2 events cleared
    assert len(l1_session.layer2_events) == 0
    
    # Speech events still have transcriptions
    speech_events = l1_session.speech_events
    assert len(speech_events) > 0
    assert speech_events[0].transcription is not None


# =============================================================================
# TESTS: Layer 2 No PII
# =============================================================================

def test_layer2_events_no_quoted_speech(sample_session):
    """Layer 2 events should not contain quoted speech."""
    for event in sample_session.layer2_events:
        result = validate_layer2_no_pii(event)
        assert result, f"Event {event.event_id} contains PII: {event}"


def test_behavioral_event_no_pii(sample_session):
    """Behavioral event descriptions should not include quoted speech."""
    behavioral_events = sample_session.behavioral_events
    assert len(behavioral_events) > 0
    
    event = behavioral_events[0]
    description = event.description or ""
    trigger_desc = event.trigger_description or ""
    
    # Check for quote marks
    assert '"' not in description, f"Quoted speech in description: {description}"
    assert "'" not in description, f"Quoted speech in description: {description}"
    assert '"' not in trigger_desc, f"Quoted speech in trigger_description: {trigger_desc}"
    assert "'" not in trigger_desc, f"Quoted speech in trigger_description: {trigger_desc}"


def test_interaction_event_no_pii(sample_session):
    """Interaction event descriptions should not include quoted speech."""
    interaction_events = sample_session.interaction_events
    assert len(interaction_events) > 0
    
    event = interaction_events[0]
    description = event.description or ""
    
    assert '"' not in description, f"Quoted speech in description: {description}"
    assert "'" not in description, f"Quoted speech in description: {description}"


def test_context_event_no_pii(sample_session):
    """Context event descriptions should not include quoted speech."""
    context_events = sample_session.context_events
    assert len(context_events) > 0
    
    event = context_events[0]
    description = event.activity_description or ""
    
    assert '"' not in description, f"Quoted speech in description: {description}"
    assert "'" not in description, f"Quoted speech in description: {description}"


# =============================================================================
# TESTS: Anonymized Session (Layer 2 Only)
# =============================================================================

def test_anonymized_session_no_layer1(sample_session):
    """Anonymized session should strip all Layer 1 events."""
    anon_session = anonymized_session(sample_session)
    
    # Layer 2 events preserved
    assert len(anon_session.layer2_events) > 0
    # All Layer 1 events cleared
    assert len(anon_session.speech_events) == 0
    assert len(anon_session.ambient_audio_events) == 0
    assert len(anon_session.proximity_events) == 0
    assert len(anon_session.gaze_events) == 0
    assert len(anon_session.posture_events) == 0
    assert len(anon_session.object_events) == 0


def test_anonymized_session_preserves_layer2(sample_session):
    """Anonymized session should preserve all Layer 2 events."""
    anon_session = anonymized_session(sample_session)
    
    # Count Layer 2 events
    original_l2_count = len(sample_session.layer2_events)
    anon_l2_count = len(anon_session.layer2_events)
    
    assert original_l2_count > 0
    assert original_l2_count == anon_l2_count


def test_anonymized_session_has_no_transcriptions(sample_session):
    """Anonymized session should have no speech transcriptions."""
    anon_session = anonymized_session(sample_session)
    
    # Check no speech events
    assert len(anon_session.speech_events) == 0
    
    # Verify Layer 2 events are intact and PII-free
    for event in anon_session.layer2_events:
        assert validate_layer2_no_pii(event)


def test_anonymized_session_preserves_metadata(sample_session):
    """Anonymized session should preserve session metadata."""
    anon_session = anonymized_session(sample_session)
    
    assert anon_session.metadata.session_id == sample_session.metadata.session_id
    assert anon_session.metadata.scenario_name == sample_session.metadata.scenario_name
    assert anon_session.metadata.num_children == sample_session.metadata.num_children
    assert anon_session.metadata.num_adults == sample_session.metadata.num_adults


# =============================================================================
# TESTS: Layer Separation Guarantees
# =============================================================================

def test_layer2_events_reference_only_event_ids(sample_session):
    """Layer 2 events should reference Layer 1 only via event IDs, not content."""
    for event in sample_session.layer2_events:
        # Check source_event_ids are proper IDs
        for event_id in event.source_event_ids:
            assert event_id.startswith("L1_"), f"Expected L1 event ID, got: {event_id}"
            assert len(event_id) > 0


def test_layer2_uses_anonymized_actor_ids(sample_session):
    """Layer 2 events should use only anonymized actor IDs (child_N, adult_N)."""
    for event in sample_session.layer2_events:
        if hasattr(event, 'actor'):
            actor_id = event.actor.id
            assert actor_id.startswith(('child_', 'adult_')), f"Non-anonymized actor ID: {actor_id}"
        if hasattr(event, 'initiator'):
            actor_id = event.initiator.id
            assert actor_id.startswith(('child_', 'adult_')), f"Non-anonymized actor ID: {actor_id}"
        if hasattr(event, 'recipient'):
            actor_id = event.recipient.id
            assert actor_id.startswith(('child_', 'adult_')), f"Non-anonymized actor ID: {actor_id}"


def test_layer1_and_layer2_are_independent(sample_session):
    """Removing Layer 1 should not affect Layer 2 content."""
    original_l2 = sample_session.layer2_events
    original_l2_ids = [e.event_id for e in original_l2]
    
    l1_only = layer1_only_session(sample_session)
    anon = anonymized_session(sample_session)
    
    # Layer 2 events in anon should match original
    anon_l2_ids = [e.event_id for e in anon.layer2_events]
    assert set(original_l2_ids) == set(anon_l2_ids)


# =============================================================================
# EDGE CASES
# =============================================================================

def test_validate_layer2_with_malformed_pii(sample_session):
    """Validation should detect various PII patterns."""
    # Create a behavioral event with quoted speech
    bad_event = BehavioralEvent(
        event_id="L2_bad_001",
        timestamp=datetime.now(),
        actor=sample_session.actors[0],
        category=BehaviorCategory.EMOTIONAL_EXPRESSION,
        description='child said "I am happy"',  # Quoted speech - should fail
        intensity=Intensity.LOW,
        apparent_emotion=EmotionalState.HAPPY,
    )
    
    assert not validate_layer2_no_pii(bad_event)


def test_empty_session_anonymization():
    """Anonymizing an empty session should work."""
    empty_session = Session(
        metadata=SessionMetadata(
            session_id="empty",
            start_time=datetime.now(),
            num_children=0,
            num_adults=0,
        ),
    )
    
    anon = anonymized_session(empty_session)
    assert len(anon.layer1_events) == 0
    assert len(anon.layer2_events) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
