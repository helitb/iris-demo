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
from .llm import (
    LLMClient,
    Config,
    get_config,
    get_available_models,
    AISUITE_AVAILABLE,
)

# Generator
from .generator import (
    Scenario,
    EventGenerator,
    load_scenario,
    load_scenarios_from_directory,
)

# Storage
from .storage import (
    serialize_session,
    deserialize_session,
    save_session,
    load_session,
    list_saved_sessions,
)

# Reports
from .reports import (
    generate_report,
    format_events_for_report,
    REPORT_TYPES,
)

# Schema
from .schema import (
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
    "Scenario", "EventGenerator", 
    "load_scenario", "load_scenarios_from_directory",
    
    # Storage
    "serialize_session", "deserialize_session",
    "save_session", "load_session", "list_saved_sessions",
    
    # Reports
    "generate_report", "format_events_for_report", "REPORT_TYPES",
    
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
