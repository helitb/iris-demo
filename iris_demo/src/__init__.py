"""
IRIS - Intelligent Room Insight System

Privacy-first behavioral observation for autism support classrooms.

Architecture:
    Layer 1: Raw sensor events from ASR/CV models
    Layer 2: LLM-inferred compound events with semantic meaning

Modules:
    - schema: Event data types
    - llm: LLM client for multi-provider access
    - generator: Event generation from scenarios
    - storage: Session persistence
    - reports: Report generation
"""

# LLM client and config
from .core.llm import (
    LLMClient,
    Config,
    get_config,
    get_available_models,
    AISUITE_AVAILABLE,
)

# Generator
from .core.scenario import (
    Scenario,
    load_scenario,
    load_scenarios_from_directory,
)
from .core import ObservationEngine

# Storage
from .core.storage import (
    serialize_session,
    deserialize_session,
    save_session,
    load_session,
    list_saved_sessions,
    save_l2_event_log,
    load_l2_event_log,
    save_validation_report,
)

# Reports
from .core.reports import (
    format_events_for_report,
)

# Reconstruction
from .core.layer1 import L1EventLog, load_l1_log_from_session
from .core.privacy import (
    PrivacyValidationReport,
    validate_layer2_no_pii,
    validate_l2_events,
    anonymized_session,
    layer1_only_session,
)

# Schema
from .core.schema import (
    # Common
    ActorRole, ClassroomZone, Intensity,
    Actor, Location, EventBase,
    
    # Layer 1 - Audio
    SpeechTarget, VocalType, VerbalComplexity,
    ProsodyFeatures, SpeechEvent, AmbientAudioEvent,
    
    # Layer 1 - Video  
    ProximityChange, ProximityLevel, ProximityEvent,
    GazeDirection, GazeEvent,
    PostureType, MovementType, BodyOrientation, PostureEvent,
    ObjectAction, ObjectEvent,
    
    # Layer 2 - LLM Inferred
    ContentType, CommunicativeIntent, BehaviorCategory,
    EmotionalState, TriggerType,
    BehavioralEvent, InteractionEvent, ContextEvent,
    InteractionType, InteractionQuality,
    ActivityType, ClassroomClimate,
    
    # Session
    SessionMetadata, Session,
    
    # Type aliases
    Layer1Event, Layer2Event, Event,
)

__version__ = "3.1.0"
__all__ = [
    # LLM
    "LLMClient", "Config", "get_config", "get_available_models", "AISUITE_AVAILABLE",
    
    # Generator
    "Scenario", "ObservationEngine",
    "load_scenario", "load_scenarios_from_directory",
    
    # Storage
    "serialize_session", "deserialize_session",
    "save_session", "load_session", "list_saved_sessions",
    
    # Reports
    "format_events_for_report",
    
    # Reconstruction
    "L1EventLog", "load_l1_log_from_session",
    "PrivacyValidationReport",
    "validate_layer2_no_pii",
    "save_l2_event_log", "load_l2_event_log",
    "validate_l2_events", "save_validation_report",
    "anonymized_session", "layer1_only_session",
    
    # Common
    "ActorRole", "ClassroomZone", "Intensity",
    "Actor", "Location", "EventBase",
    
    # Layer 1 - Audio
    "SpeechTarget", "VocalType", "VerbalComplexity", 
    "ProsodyFeatures", "SpeechEvent", "AmbientAudioEvent",
    
    # Layer 1 - Video
    "ProximityChange", "ProximityLevel", "ProximityEvent",
    "GazeDirection", "GazeEvent",
    "PostureType", "MovementType", "BodyOrientation", "PostureEvent",
    "ObjectAction", "ObjectEvent",
    
    # Layer 2 - LLM Inferred
    "ContentType", "CommunicativeIntent", "BehaviorCategory",
    "EmotionalState", "TriggerType",
    "BehavioralEvent", "InteractionEvent", "ContextEvent",
    "InteractionType", "InteractionQuality",
    "ActivityType", "ClassroomClimate",
    
    # Session
    "SessionMetadata", "Session",
    
    # Type aliases
    "Layer1Event", "Layer2Event", "Event",
]
