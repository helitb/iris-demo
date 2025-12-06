"""
IRIS Session Serialization

Save and load sessions to/from JSON for persistence and review.
"""

import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import asdict

from .schema import (
    Actor, ActorRole, Location, ClassroomZone, Intensity,
    Session, SessionMetadata,
    
    # Layer 1
    SpeechEvent, AmbientAudioEvent, ProximityEvent, GazeEvent,
    PostureEvent, ObjectEvent,
    ProsodyFeatures, SpeechTarget, VocalType, VerbalComplexity,
    ProximityChange, ProximityLevel,
    GazeDirection,
    PostureType, MovementType, BodyOrientation,
    ObjectAction,
    
    # Layer 2
    BehavioralEvent, InteractionEvent, ContextEvent,
    ContentType, CommunicativeIntent, BehaviorCategory,
    EmotionalState, TriggerType,
    InteractionType, InteractionQuality,
    ActivityType, ClassroomClimate,
)


# =============================================================================
# SERIALIZATION
# =============================================================================

def _serialize_datetime(dt: datetime) -> str:
    return dt.isoformat() if dt else None


def _serialize_enum(val) -> Optional[str]:
    return val.value if val else None


def _serialize_actor(actor: Optional[Actor]) -> Optional[Dict]:
    if not actor:
        return None
    return {"id": actor.id, "role": actor.role.value}


def _serialize_location(loc: Optional[Location]) -> Optional[Dict]:
    if not loc:
        return None
    return {
        "zone": _serialize_enum(loc.zone),
        "x": loc.x,
        "y": loc.y,
    }


def _serialize_prosody(prosody: Optional[ProsodyFeatures]) -> Optional[Dict]:
    if not prosody:
        return None
    return {
        "pitch_mean_hz": prosody.pitch_mean_hz,
        "pitch_std_hz": prosody.pitch_std_hz,
        "pitch_contour": prosody.pitch_contour,
        "intensity_mean_db": prosody.intensity_mean_db,
        "intensity_range_db": prosody.intensity_range_db,
        "speech_rate": prosody.speech_rate,
        "rhythm_regularity": prosody.rhythm_regularity,
        "voice_quality": prosody.voice_quality,
    }


def serialize_event(event) -> Dict[str, Any]:
    """Serialize any event to a dictionary."""
    base = {
        "event_id": event.event_id,
        "timestamp": _serialize_datetime(event.timestamp),
    }
    
    if isinstance(event, SpeechEvent):
        return {
            **base,
            "_type": "SpeechEvent",
            "speaker": _serialize_actor(event.speaker),
            "transcription": event.transcription,
            "word_count": event.word_count,
            "complexity": _serialize_enum(event.complexity),
            "vocal_type": _serialize_enum(event.vocal_type),
            "target": _serialize_enum(event.target),
            "prosody": _serialize_prosody(event.prosody),
            "duration_ms": event.duration_ms,
            "gap_before_ms": event.gap_before_ms,
            "is_overlap": event.is_overlap,
            "previous_speaker": event.previous_speaker,
            "is_echolalia_candidate": event.is_echolalia_candidate,
            "echolalia_similarity": event.echolalia_similarity,
            "echolalia_delay_ms": event.echolalia_delay_ms,
            "is_perseveration_candidate": event.is_perseveration_candidate,
        }
    
    elif isinstance(event, AmbientAudioEvent):
        return {
            **base,
            "_type": "AmbientAudioEvent",
            "sound_type": event.sound_type,
            "intensity": _serialize_enum(event.intensity),
            "duration_ms": event.duration_ms,
            "location_estimate": _serialize_enum(event.location_estimate),
        }
    
    elif isinstance(event, ProximityEvent):
        return {
            **base,
            "_type": "ProximityEvent",
            "actor": _serialize_actor(event.actor),
            "target": _serialize_actor(event.target),
            "change_type": _serialize_enum(event.change_type),
            "proximity_level": _serialize_enum(event.proximity_level),
            "distance_meters": event.distance_meters,
            "movement_speed": event.movement_speed,
            "actor_location": _serialize_location(event.actor_location),
        }
    
    elif isinstance(event, GazeEvent):
        return {
            **base,
            "_type": "GazeEvent",
            "actor": _serialize_actor(event.actor),
            "direction": _serialize_enum(event.direction),
            "target_actor": _serialize_actor(event.target_actor),
            "target_object": event.target_object,
            "target_zone": _serialize_enum(event.target_zone),
            "duration_ms": event.duration_ms,
            "is_mutual": event.is_mutual,
            "is_fleeting": event.is_fleeting,
            "is_sustained": event.is_sustained,
        }
    
    elif isinstance(event, PostureEvent):
        return {
            **base,
            "_type": "PostureEvent",
            "actor": _serialize_actor(event.actor),
            "posture": _serialize_enum(event.posture),
            "movement": _serialize_enum(event.movement),
            "orientation": _serialize_enum(event.orientation),
            "location": _serialize_location(event.location),
            "movement_intensity": _serialize_enum(event.movement_intensity),
            "is_repetitive": event.is_repetitive,
            "repetition_frequency": event.repetition_frequency,
        }
    
    elif isinstance(event, ObjectEvent):
        return {
            **base,
            "_type": "ObjectEvent",
            "actor": _serialize_actor(event.actor),
            "object_type": event.object_type,
            "action": _serialize_enum(event.action),
            "is_appropriate": event.is_appropriate,
            "shared_with": _serialize_actor(event.shared_with),
        }
    
    elif isinstance(event, BehavioralEvent):
        return {
            **base,
            "_type": "BehavioralEvent",
            "actor": _serialize_actor(event.actor),
            "category": _serialize_enum(event.category),
            "description": event.description,
            "intensity": _serialize_enum(event.intensity),
            "apparent_emotion": _serialize_enum(event.apparent_emotion),
            "trigger": _serialize_enum(event.trigger),
            "trigger_description": event.trigger_description,
            "regulation_effective": event.regulation_effective,
            "source_event_ids": event.source_event_ids,
            "confidence": event.confidence,
        }
    
    elif isinstance(event, InteractionEvent):
        return {
            **base,
            "_type": "InteractionEvent",
            "initiator": _serialize_actor(event.initiator),
            "recipient": _serialize_actor(event.recipient),
            "interaction_type": _serialize_enum(event.interaction_type),
            "description": event.description,
            "quality": _serialize_enum(event.quality),
            "reciprocity_level": event.reciprocity_level,
            "content_type": _serialize_enum(event.content_type),
            "communicative_intent": _serialize_enum(event.communicative_intent),
            "adult_facilitated": event.adult_facilitated,
            "adult_actor": _serialize_actor(event.adult_actor),
            "source_event_ids": event.source_event_ids,
            "confidence": event.confidence,
        }
    
    elif isinstance(event, ContextEvent):
        return {
            **base,
            "_type": "ContextEvent",
            "activity_type": _serialize_enum(event.activity_type),
            "activity_description": event.activity_description,
            "primary_zone": _serialize_enum(event.primary_zone),
            "classroom_climate": _serialize_enum(event.classroom_climate),
            "noise_level": _serialize_enum(event.noise_level),
            "is_transition": event.is_transition,
            "transition_from": _serialize_enum(event.transition_from),
            "transition_to": _serialize_enum(event.transition_to),
            "adults_present": event.adults_present,
            "adult_attention_focus": event.adult_attention_focus,
            "source_event_ids": event.source_event_ids,
            "confidence": event.confidence,
        }
    
    return base


def serialize_session(session: Session) -> Dict[str, Any]:
    """Serialize a complete session to a dictionary."""
    return {
        "metadata": {
            "session_id": session.metadata.session_id,
            "start_time": _serialize_datetime(session.metadata.start_time),
            "end_time": _serialize_datetime(session.metadata.end_time),
            "scenario_name": session.metadata.scenario_name,
            "scenario_description": session.metadata.scenario_description,
            "num_children": session.metadata.num_children,
            "num_adults": session.metadata.num_adults,
            "layer1_model_versions": session.metadata.layer1_model_versions,
            "layer2_llm_model": session.metadata.layer2_llm_model,
        },
        "actors": [_serialize_actor(a) for a in session.actors],
        "layer1_events": [serialize_event(e) for e in session.layer1_events],
        "layer2_events": [serialize_event(e) for e in session.layer2_events],
    }


# =============================================================================
# DESERIALIZATION
# =============================================================================

def _parse_datetime(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s)


def _parse_actor(data: Optional[Dict]) -> Optional[Actor]:
    if not data:
        return None
    return Actor(id=data["id"], role=ActorRole(data["role"]))


def _parse_location(data: Optional[Dict]) -> Optional[Location]:
    if not data:
        return None
    return Location(
        zone=ClassroomZone(data["zone"]) if data.get("zone") else ClassroomZone.UNKNOWN,
        x=data.get("x"),
        y=data.get("y"),
    )


def _parse_prosody(data: Optional[Dict]) -> Optional[ProsodyFeatures]:
    if not data:
        return None
    return ProsodyFeatures(
        pitch_mean_hz=data.get("pitch_mean_hz"),
        pitch_std_hz=data.get("pitch_std_hz"),
        pitch_contour=data.get("pitch_contour"),
        intensity_mean_db=data.get("intensity_mean_db"),
        intensity_range_db=data.get("intensity_range_db"),
        speech_rate=data.get("speech_rate"),
        rhythm_regularity=data.get("rhythm_regularity"),
        voice_quality=data.get("voice_quality"),
    )


def _safe_enum(enum_class, value, default=None):
    if value is None:
        return default
    try:
        return enum_class(value)
    except ValueError:
        return default


def deserialize_event(data: Dict[str, Any]):
    """Deserialize an event from a dictionary."""
    event_type = data.get("_type")
    timestamp = _parse_datetime(data["timestamp"])
    event_id = data["event_id"]
    
    if event_type == "SpeechEvent":
        return SpeechEvent(
            event_id=event_id,
            timestamp=timestamp,
            speaker=_parse_actor(data["speaker"]),
            transcription=data.get("transcription"),
            word_count=data.get("word_count", 0),
            complexity=_safe_enum(VerbalComplexity, data.get("complexity"), VerbalComplexity.VOCALIZATION),
            vocal_type=_safe_enum(VocalType, data.get("vocal_type"), VocalType.SPEECH),
            target=_safe_enum(SpeechTarget, data.get("target"), SpeechTarget.UNKNOWN),
            prosody=_parse_prosody(data.get("prosody")),
            duration_ms=data.get("duration_ms", 0),
            gap_before_ms=data.get("gap_before_ms"),
            is_overlap=data.get("is_overlap", False),
            previous_speaker=data.get("previous_speaker"),
            is_echolalia_candidate=data.get("is_echolalia_candidate", False),
            echolalia_similarity=data.get("echolalia_similarity"),
            echolalia_delay_ms=data.get("echolalia_delay_ms"),
            is_perseveration_candidate=data.get("is_perseveration_candidate", False),
        )
    
    elif event_type == "AmbientAudioEvent":
        return AmbientAudioEvent(
            event_id=event_id,
            timestamp=timestamp,
            sound_type=data.get("sound_type", "unknown"),
            intensity=_safe_enum(Intensity, data.get("intensity"), Intensity.MODERATE),
            duration_ms=data.get("duration_ms", 0),
            location_estimate=_safe_enum(ClassroomZone, data.get("location_estimate")),
        )
    
    elif event_type == "ProximityEvent":
        return ProximityEvent(
            event_id=event_id,
            timestamp=timestamp,
            actor=_parse_actor(data["actor"]),
            target=_parse_actor(data["target"]),
            change_type=_safe_enum(ProximityChange, data.get("change_type"), ProximityChange.STABLE),
            proximity_level=_safe_enum(ProximityLevel, data.get("proximity_level"), ProximityLevel.SOCIAL),
            distance_meters=data.get("distance_meters"),
            movement_speed=data.get("movement_speed"),
            actor_location=_parse_location(data.get("actor_location")),
        )
    
    elif event_type == "GazeEvent":
        return GazeEvent(
            event_id=event_id,
            timestamp=timestamp,
            actor=_parse_actor(data["actor"]),
            direction=_safe_enum(GazeDirection, data.get("direction"), GazeDirection.UNFOCUSED),
            target_actor=_parse_actor(data.get("target_actor")),
            target_object=data.get("target_object"),
            target_zone=_safe_enum(ClassroomZone, data.get("target_zone")),
            duration_ms=data.get("duration_ms", 0),
            is_mutual=data.get("is_mutual", False),
            is_fleeting=data.get("is_fleeting", False),
            is_sustained=data.get("is_sustained", False),
        )
    
    elif event_type == "PostureEvent":
        return PostureEvent(
            event_id=event_id,
            timestamp=timestamp,
            actor=_parse_actor(data["actor"]),
            posture=_safe_enum(PostureType, data.get("posture"), PostureType.SITTING),
            movement=_safe_enum(MovementType, data.get("movement"), MovementType.STILL),
            orientation=_safe_enum(BodyOrientation, data.get("orientation"), BodyOrientation.ENGAGED),
            location=_parse_location(data.get("location")),
            movement_intensity=_safe_enum(Intensity, data.get("movement_intensity"), Intensity.LOW),
            is_repetitive=data.get("is_repetitive", False),
            repetition_frequency=data.get("repetition_frequency"),
        )
    
    elif event_type == "ObjectEvent":
        return ObjectEvent(
            event_id=event_id,
            timestamp=timestamp,
            actor=_parse_actor(data["actor"]),
            object_type=data.get("object_type", "unknown"),
            action=_safe_enum(ObjectAction, data.get("action"), ObjectAction.MANIPULATE),
            is_appropriate=data.get("is_appropriate"),
            shared_with=_parse_actor(data.get("shared_with")),
        )
    
    elif event_type == "BehavioralEvent":
        return BehavioralEvent(
            event_id=event_id,
            timestamp=timestamp,
            actor=_parse_actor(data["actor"]),
            category=_safe_enum(BehaviorCategory, data.get("category"), BehaviorCategory.EMOTIONAL_EXPRESSION),
            description=data.get("description", ""),
            intensity=_safe_enum(Intensity, data.get("intensity"), Intensity.MODERATE),
            apparent_emotion=_safe_enum(EmotionalState, data.get("apparent_emotion"), EmotionalState.UNCLEAR),
            trigger=_safe_enum(TriggerType, data.get("trigger"), TriggerType.UNKNOWN),
            trigger_description=data.get("trigger_description"),
            regulation_effective=data.get("regulation_effective"),
            source_event_ids=data.get("source_event_ids", []),
            confidence=data.get("confidence", 0.8),
        )
    
    elif event_type == "InteractionEvent":
        return InteractionEvent(
            event_id=event_id,
            timestamp=timestamp,
            initiator=_parse_actor(data["initiator"]),
            recipient=_parse_actor(data["recipient"]),
            interaction_type=_safe_enum(InteractionType, data.get("interaction_type"), InteractionType.INITIATION),
            description=data.get("description", ""),
            quality=_safe_enum(InteractionQuality, data.get("quality"), InteractionQuality.ONGOING),
            reciprocity_level=data.get("reciprocity_level"),
            content_type=_safe_enum(ContentType, data.get("content_type")),
            communicative_intent=_safe_enum(CommunicativeIntent, data.get("communicative_intent")),
            adult_facilitated=data.get("adult_facilitated", False),
            adult_actor=_parse_actor(data.get("adult_actor")),
            source_event_ids=data.get("source_event_ids", []),
            confidence=data.get("confidence", 0.8),
        )
    
    elif event_type == "ContextEvent":
        return ContextEvent(
            event_id=event_id,
            timestamp=timestamp,
            activity_type=_safe_enum(ActivityType, data.get("activity_type"), ActivityType.FREE_PLAY),
            activity_description=data.get("activity_description"),
            primary_zone=_safe_enum(ClassroomZone, data.get("primary_zone"), ClassroomZone.UNKNOWN),
            classroom_climate=_safe_enum(ClassroomClimate, data.get("classroom_climate"), ClassroomClimate.CALM),
            noise_level=_safe_enum(Intensity, data.get("noise_level"), Intensity.MODERATE),
            is_transition=data.get("is_transition", False),
            transition_from=_safe_enum(ActivityType, data.get("transition_from")),
            transition_to=_safe_enum(ActivityType, data.get("transition_to")),
            adults_present=data.get("adults_present", 1),
            adult_attention_focus=data.get("adult_attention_focus"),
            source_event_ids=data.get("source_event_ids", []),
            confidence=data.get("confidence", 0.8),
        )
    
    raise ValueError(f"Unknown event type: {event_type}")


def deserialize_session(data: Dict[str, Any]) -> Session:
    """Deserialize a session from a dictionary."""
    meta = data["metadata"]
    
    session = Session(
        metadata=SessionMetadata(
            session_id=meta["session_id"],
            start_time=_parse_datetime(meta["start_time"]),
            end_time=_parse_datetime(meta.get("end_time")),
            scenario_name=meta.get("scenario_name"),
            scenario_description=meta.get("scenario_description"),
            num_children=meta.get("num_children", 0),
            num_adults=meta.get("num_adults", 0),
            layer1_model_versions=meta.get("layer1_model_versions"),
            layer2_llm_model=meta.get("layer2_llm_model"),
        ),
        actors=[_parse_actor(a) for a in data.get("actors", [])],
    )
    
    # Deserialize and categorize events
    for event_data in data.get("layer1_events", []):
        event = deserialize_event(event_data)
        if isinstance(event, SpeechEvent):
            session.speech_events.append(event)
        elif isinstance(event, AmbientAudioEvent):
            session.ambient_audio_events.append(event)
        elif isinstance(event, ProximityEvent):
            session.proximity_events.append(event)
        elif isinstance(event, GazeEvent):
            session.gaze_events.append(event)
        elif isinstance(event, PostureEvent):
            session.posture_events.append(event)
        elif isinstance(event, ObjectEvent):
            session.object_events.append(event)
    
    for event_data in data.get("layer2_events", []):
        event = deserialize_event(event_data)
        if isinstance(event, BehavioralEvent):
            session.behavioral_events.append(event)
        elif isinstance(event, InteractionEvent):
            session.interaction_events.append(event)
        elif isinstance(event, ContextEvent):
            session.context_events.append(event)
    
    return session


# =============================================================================
# FILE I/O
# =============================================================================

def save_session(session: Session, path: str) -> None:
    """Save a session to a JSON file."""
    data = serialize_session(session)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_session(path: str) -> Session:
    """Load a session from a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return deserialize_session(data)


def list_saved_sessions(directory: str) -> List[Dict[str, Any]]:
    """List all saved sessions in a directory with basic metadata."""
    sessions = []
    
    if not os.path.exists(directory):
        return sessions
    
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            path = os.path.join(directory, filename)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                meta = data.get("metadata", {})
                sessions.append({
                    "filename": filename,
                    "path": path,
                    "session_id": meta.get("session_id"),
                    "scenario_name": meta.get("scenario_name"),
                    "start_time": meta.get("start_time"),
                    "num_children": meta.get("num_children"),
                    "layer1_count": len(data.get("layer1_events", [])),
                    "layer2_count": len(data.get("layer2_events", [])),
                })
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return sorted(sessions, key=lambda s: s.get("start_time", ""), reverse=True)
