"""
IRIS Event Generator v3.1

LLM-based event generation pipeline:
  Step 1: Scenario text → Layer 1 events (raw sensor simulation)
  Step 2: Layer 1 events → Layer 2 events (semantic inference)

Uses the LLM module for unified LLM access.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Generator, Optional, List, Dict, Any, Callable
from dataclasses import dataclass

from .llm import LLMClient, get_config
from .schema import (
    # Common
    Actor, ActorRole, Location, ClassroomZone, Intensity,
    
    # Layer 1
    SpeechEvent, AmbientAudioEvent, ProximityEvent, GazeEvent, 
    PostureEvent, ObjectEvent,
    SpeechTarget, VocalType, VerbalComplexity, ProsodyFeatures,
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
    
    # Session
    Session, SessionMetadata,
    Layer1Event, Layer2Event,
)


# =============================================================================
# SCENARIO SCHEMA
# =============================================================================

@dataclass
class Scenario:
    """Scenario definition loaded from JSON."""
    name: str
    description: str
    duration_minutes: int = 10
    
    # Classroom setup
    num_children: int = 6
    num_adults: int = 2
    
    # Optional: specific child profiles to highlight
    focus_children: Optional[List[Dict[str, Any]]] = None
    
    # Optional: key moments to ensure are captured
    key_moments: Optional[List[Dict[str, str]]] = None
    
    @classmethod
    def from_json(cls, path: str) -> "Scenario":
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Scenario":
        return cls(**data)
    
    @classmethod
    def from_text(cls, name: str, description: str, **kwargs) -> "Scenario":
        """Create scenario from free-form text description."""
        return cls(name=name, description=description, **kwargs)


# =============================================================================
# PROMPTS
# =============================================================================

LAYER1_SYSTEM_PROMPT = """You are simulating raw sensor data from a classroom observation system for autism support.

Your task: Given a scenario description, generate a stream of Layer 1 (raw sensor) events that would be detected by:
- Audio sensors: speech detection, ambient sounds
- Video sensors: proximity tracking, gaze detection, posture/movement, object interactions

IMPORTANT RULES:
1. Generate events in chronological order
2. Use anonymous IDs only (child_1, child_2, adult_1, etc.)
3. Events should be plausible raw sensor output - no interpretations yet
4. Include realistic timing gaps between events
5. Speech transcriptions should be in Hebrew when children/teachers speak
6. Generate diverse event types - don't over-focus on just speech

OUTPUT FORMAT:
Output each event as a JSON object on its own line, prefixed with the event type.
Do NOT wrap in markdown code blocks.

Example output format:
SPEECH|{"speaker_id": "child_1", "timestamp_offset_sec": 0, "transcription": "אני רוצה את הכדור", "word_count": 4, "vocal_type": "speech", "target": "peer", "duration_ms": 1200}
AMBIENT|{"timestamp_offset_sec": 5, "sound_type": "door", "intensity": "moderate", "duration_ms": 800}
PROXIMITY|{"actor_id": "child_2", "target_id": "child_1", "timestamp_offset_sec": 8, "change_type": "approach", "proximity_level": "personal", "zone": "play_area"}
GAZE|{"actor_id": "child_1", "timestamp_offset_sec": 10, "direction": "at_person", "target_actor_id": "child_2", "duration_ms": 500, "is_mutual": false}
POSTURE|{"actor_id": "child_3", "timestamp_offset_sec": 12, "posture": "sitting", "movement": "rocking", "orientation": "turned_away", "movement_intensity": "moderate", "is_repetitive": true}
OBJECT|{"actor_id": "child_1", "timestamp_offset_sec": 15, "object_type": "ball", "action": "give", "shared_with_id": "child_2"}

EVENT TYPES AND FIELDS:

SPEECH - Speech detection from ASR
- speaker_id: string (child_N or adult_N)
- timestamp_offset_sec: number
- transcription: string (Hebrew for actual speech, null for non-speech vocalizations)
- word_count: number
- complexity: "vocalization" | "single_word" | "phrase" | "sentence" | "multi_sentence"
- vocal_type: "speech" | "vocalization" | "cry" | "laugh" | "scream" | "hum" | "whisper"
- target: "peer" | "adult" | "self" | "broadcast" | "unknown"
- duration_ms: number
- prosody: {"pitch_contour": "rising|falling|flat|variable", "speech_rate": "slow|normal|fast", "intensity_mean_db": number} (optional)
- gap_before_ms: number (optional, silence before utterance)
- is_overlap: boolean (optional)
- is_echolalia_candidate: boolean (optional, if repeating recent utterance)
- echolalia_similarity: number 0-1 (optional)

AMBIENT - Environmental audio
- timestamp_offset_sec: number
- sound_type: string (bell, door, chair_scrape, crash, music, drilling, siren, etc.)
- intensity: "low" | "moderate" | "high"
- duration_ms: number
- zone: string (optional, estimated location)

PROXIMITY - Spatial tracking
- actor_id: string
- target_id: string
- timestamp_offset_sec: number
- change_type: "approach" | "withdrawal" | "stable"
- proximity_level: "intimate" | "personal" | "social" | "public"
- distance_meters: number (optional)
- movement_speed: "slow" | "normal" | "fast" | "abrupt" (optional)
- zone: string

GAZE - Eye tracking
- actor_id: string
- timestamp_offset_sec: number
- direction: "at_person" | "at_object" | "at_activity" | "averted" | "downcast" | "scanning" | "unfocused"
- target_actor_id: string (optional, if looking at person)
- target_object: string (optional, if looking at object)
- target_zone: string (optional)
- duration_ms: number
- is_mutual: boolean
- is_fleeting: boolean (optional)
- is_sustained: boolean (optional)

POSTURE - Body position/movement
- actor_id: string
- timestamp_offset_sec: number
- posture: "standing" | "sitting" | "kneeling" | "lying" | "crouching"
- movement: "still" | "walking" | "running" | "jumping" | "rocking" | "pacing" | "hand_flapping" | "spinning" | "fidgeting" | "reaching" | "pointing"
- orientation: "engaged" | "partially_turned" | "turned_away" | "back_to_group"
- movement_intensity: "low" | "moderate" | "high"
- is_repetitive: boolean
- zone: string

OBJECT - Object interactions
- actor_id: string
- timestamp_offset_sec: number
- object_type: string (toy, book, pencil, fidget, food, blocks, puzzle, etc.)
- action: "pick_up" | "put_down" | "hold" | "manipulate" | "throw" | "push" | "give" | "receive" | "point_at"
- shared_with_id: string (optional)
- is_appropriate: boolean (optional)

Generate a realistic stream of 40-80 events for the scenario. Space them naturally across the duration.
Focus on observable behaviors, not interpretations."""


LAYER2_SYSTEM_PROMPT = """You are an AI system that infers high-level behavioral and social meaning from raw sensor data in an autism support classroom.

Your task: Given a stream of Layer 1 (raw sensor) events, generate Layer 2 (interpreted) events that capture:
- BehavioralEvent: Meaningful behaviors with triggers and emotional states
- InteractionEvent: Social exchanges and their quality
- ContextEvent: Classroom climate and activity phases

IMPORTANT RULES:
1. Infer meaning from PATTERNS across multiple Layer 1 events
2. Reference source events by their IDs
3. Generate ContextEvents periodically (every 30-60 seconds of scenario time)
4. Be clinically relevant - focus on behaviors that matter for intervention planning
5. Don't over-interpret - use "unclear" when uncertain

OUTPUT FORMAT:
Output each event as a JSON object on its own line, prefixed with the event type.
Do NOT wrap in markdown code blocks.

Example output format:
BEHAVIORAL|{"actor_id": "child_3", "timestamp_offset_sec": 45, "category": "self_regulation", "description": "rocking and covering ears in response to noise", "intensity": "high", "apparent_emotion": "overwhelmed", "trigger": "sensory_input", "trigger_description": "drilling noise from hallway", "source_event_ids": ["evt_12", "evt_15", "evt_18"], "confidence": 0.85}
INTERACTION|{"initiator_id": "child_1", "recipient_id": "child_2", "timestamp_offset_sec": 60, "interaction_type": "initiation", "description": "offered toy and made eye contact", "quality": "successful", "reciprocity_level": "moderate", "content_type": "request", "source_event_ids": ["evt_20", "evt_21", "evt_22"], "confidence": 0.9}
CONTEXT|{"timestamp_offset_sec": 90, "activity_type": "free_play", "primary_zone": "play_area", "classroom_climate": "energetic", "noise_level": "moderate", "is_transition": false, "adults_present": 2, "source_event_ids": ["evt_25", "evt_30"], "confidence": 0.95}

EVENT TYPES AND FIELDS:

BEHAVIORAL - Interpreted behaviors
- actor_id: string
- timestamp_offset_sec: number
- category: "stimming" | "self_regulation" | "sensory_seeking" | "sensory_avoidance" | "emotional_expression" | "social_approach" | "social_withdrawal" | "task_engagement" | "task_avoidance" | "distress" | "aggression" | "compliance" | "noncompliance"
- description: string (brief, clinical description)
- intensity: "low" | "moderate" | "high"
- apparent_emotion: "calm" | "happy" | "excited" | "anxious" | "frustrated" | "sad" | "angry" | "overwhelmed" | "withdrawn" | "neutral" | "unclear"
- trigger: "sensory_input" | "social_demand" | "transition" | "frustration" | "anticipation" | "unmet_need" | "peer_action" | "adult_action" | "environmental" | "internal" | "unknown"
- trigger_description: string (optional, what specifically triggered it)
- regulation_effective: boolean (optional, if self-regulation behavior)
- source_event_ids: list of strings
- confidence: number 0-1

INTERACTION - Social exchanges
- initiator_id: string
- recipient_id: string
- timestamp_offset_sec: number
- interaction_type: "initiation" | "response" | "joint_attention" | "parallel_play" | "cooperative_play" | "help_seeking" | "help_giving" | "conflict" | "repair" | "ignore" | "rejection"
- description: string
- quality: "successful" | "partial" | "unsuccessful" | "ongoing"
- reciprocity_level: "none" | "minimal" | "moderate" | "full" (optional)
- content_type: "request" | "protest" | "comment" | "response" | "question" | "greeting" | "echolalia_immediate" | "echolalia_delayed" | "self_talk" (optional)
- communicative_intent: "social_connection" | "information_seeking" | "emotional_expression" | "self_regulation" | "attention_seeking" | "need_expression" | "play_initiation" | "unclear" (optional)
- adult_facilitated: boolean
- adult_actor_id: string (optional)
- source_event_ids: list of strings
- confidence: number 0-1

CONTEXT - Classroom state (generate every 30-60 seconds)
- timestamp_offset_sec: number
- activity_type: "circle_time" | "free_play" | "structured_activity" | "snack_time" | "outdoor_play" | "transition" | "one_on_one" | "small_group" | "cleanup" | "arrival" | "departure" | "sensory_break"
- activity_description: string (optional)
- primary_zone: "circle_area" | "work_tables" | "sensory_corner" | "play_area" | "quiet_corner" | "entrance" | "bathroom_area" | "teacher_desk" | "unknown"
- classroom_climate: "calm" | "focused" | "energetic" | "restless" | "chaotic" | "tense"
- noise_level: "low" | "moderate" | "high"
- is_transition: boolean
- transition_from: string (optional, activity type)
- transition_to: string (optional, activity type)
- adults_present: number
- adult_attention_focus: string (optional, "whole group", "child_3", etc.)
- source_event_ids: list of strings
- confidence: number 0-1

Generate 15-30 Layer 2 events that capture the meaningful patterns in the Layer 1 data.
Focus on clinically relevant observations."""


# =============================================================================
# EVENT PARSING
# =============================================================================

def _safe_enum(enum_class, value, default=None):
    """Safely convert value to enum, returning default on failure."""
    if value is None:
        return default
    try:
        return enum_class(value)
    except ValueError:
        return default


def parse_layer1_event(
    event_type: str, 
    data: dict, 
    base_time: datetime, 
    actors: Dict[str, Actor]
) -> Optional[Layer1Event]:
    """Parse a Layer 1 event from LLM JSON output."""
    
    timestamp = base_time + timedelta(seconds=data.get("timestamp_offset_sec", 0))
    event_id = data.get("event_id", f"L1_{uuid.uuid4().hex[:8]}")
    
    def get_actor(actor_id: str) -> Actor:
        if actor_id not in actors:
            role = ActorRole.ADULT if actor_id.startswith("adult") else ActorRole.CHILD
            actors[actor_id] = Actor(id=actor_id, role=role)
        return actors[actor_id]
    
    try:
        if event_type == "SPEECH":
            prosody = None
            if "prosody" in data and data["prosody"]:
                prosody = ProsodyFeatures(
                    pitch_contour=data["prosody"].get("pitch_contour"),
                    intensity_mean_db=data["prosody"].get("intensity_mean_db"),
                    speech_rate=data["prosody"].get("speech_rate"),
                )
            
            return SpeechEvent(
                event_id=event_id,
                timestamp=timestamp,
                speaker=get_actor(data["speaker_id"]),
                transcription=data.get("transcription"),
                word_count=data.get("word_count", 0),
                complexity=_safe_enum(VerbalComplexity, data.get("complexity"), VerbalComplexity.VOCALIZATION),
                vocal_type=_safe_enum(VocalType, data.get("vocal_type"), VocalType.SPEECH),
                target=_safe_enum(SpeechTarget, data.get("target"), SpeechTarget.UNKNOWN),
                prosody=prosody,
                duration_ms=data.get("duration_ms", 0),
                gap_before_ms=data.get("gap_before_ms"),
                is_overlap=data.get("is_overlap", False),
                is_echolalia_candidate=data.get("is_echolalia_candidate", False),
                echolalia_similarity=data.get("echolalia_similarity"),
            )
        
        elif event_type == "AMBIENT":
            return AmbientAudioEvent(
                event_id=event_id,
                timestamp=timestamp,
                sound_type=data.get("sound_type", "unknown"),
                intensity=_safe_enum(Intensity, data.get("intensity"), Intensity.MODERATE),
                duration_ms=data.get("duration_ms", 0),
                location_estimate=_safe_enum(ClassroomZone, data.get("zone")),
            )
        
        elif event_type == "PROXIMITY":
            return ProximityEvent(
                event_id=event_id,
                timestamp=timestamp,
                actor=get_actor(data["actor_id"]),
                target=get_actor(data["target_id"]),
                change_type=_safe_enum(ProximityChange, data.get("change_type"), ProximityChange.STABLE),
                proximity_level=_safe_enum(ProximityLevel, data.get("proximity_level"), ProximityLevel.SOCIAL),
                distance_meters=data.get("distance_meters"),
                movement_speed=data.get("movement_speed"),
                actor_location=Location(zone=_safe_enum(ClassroomZone, data.get("zone"), ClassroomZone.UNKNOWN)),
            )
        
        elif event_type == "GAZE":
            return GazeEvent(
                event_id=event_id,
                timestamp=timestamp,
                actor=get_actor(data["actor_id"]),
                direction=_safe_enum(GazeDirection, data.get("direction"), GazeDirection.UNFOCUSED),
                target_actor=get_actor(data["target_actor_id"]) if data.get("target_actor_id") else None,
                target_object=data.get("target_object"),
                target_zone=_safe_enum(ClassroomZone, data.get("target_zone")),
                duration_ms=data.get("duration_ms", 0),
                is_mutual=data.get("is_mutual", False),
                is_fleeting=data.get("is_fleeting", False),
                is_sustained=data.get("is_sustained", False),
            )
        
        elif event_type == "POSTURE":
            return PostureEvent(
                event_id=event_id,
                timestamp=timestamp,
                actor=get_actor(data["actor_id"]),
                posture=_safe_enum(PostureType, data.get("posture"), PostureType.SITTING),
                movement=_safe_enum(MovementType, data.get("movement"), MovementType.STILL),
                orientation=_safe_enum(BodyOrientation, data.get("orientation"), BodyOrientation.ENGAGED),
                location=Location(zone=_safe_enum(ClassroomZone, data.get("zone"), ClassroomZone.UNKNOWN)),
                movement_intensity=_safe_enum(Intensity, data.get("movement_intensity"), Intensity.LOW),
                is_repetitive=data.get("is_repetitive", False),
                repetition_frequency=data.get("repetition_frequency"),
            )
        
        elif event_type == "OBJECT":
            return ObjectEvent(
                event_id=event_id,
                timestamp=timestamp,
                actor=get_actor(data["actor_id"]),
                object_type=data.get("object_type", "unknown"),
                action=_safe_enum(ObjectAction, data.get("action"), ObjectAction.MANIPULATE),
                is_appropriate=data.get("is_appropriate"),
                shared_with=get_actor(data["shared_with_id"]) if data.get("shared_with_id") else None,
            )
    
    except Exception as e:
        print(f"Error parsing {event_type}: {e}")
        return None
    
    return None


def parse_layer2_event(
    event_type: str, 
    data: dict, 
    base_time: datetime, 
    actors: Dict[str, Actor]
) -> Optional[Layer2Event]:
    """Parse a Layer 2 event from LLM JSON output."""
    
    timestamp = base_time + timedelta(seconds=data.get("timestamp_offset_sec", 0))
    event_id = data.get("event_id", f"L2_{uuid.uuid4().hex[:8]}")
    
    def get_actor(actor_id: str) -> Actor:
        if actor_id not in actors:
            role = ActorRole.ADULT if actor_id.startswith("adult") else ActorRole.CHILD
            actors[actor_id] = Actor(id=actor_id, role=role)
        return actors[actor_id]
    
    try:
        if event_type == "BEHAVIORAL":
            return BehavioralEvent(
                event_id=event_id,
                timestamp=timestamp,
                actor=get_actor(data["actor_id"]),
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
        
        elif event_type == "INTERACTION":
            return InteractionEvent(
                event_id=event_id,
                timestamp=timestamp,
                initiator=get_actor(data["initiator_id"]),
                recipient=get_actor(data["recipient_id"]),
                interaction_type=_safe_enum(InteractionType, data.get("interaction_type"), InteractionType.INITIATION),
                description=data.get("description", ""),
                quality=_safe_enum(InteractionQuality, data.get("quality"), InteractionQuality.ONGOING),
                reciprocity_level=data.get("reciprocity_level"),
                content_type=_safe_enum(ContentType, data.get("content_type")),
                communicative_intent=_safe_enum(CommunicativeIntent, data.get("communicative_intent")),
                adult_facilitated=data.get("adult_facilitated", False),
                adult_actor=get_actor(data["adult_actor_id"]) if data.get("adult_actor_id") else None,
                source_event_ids=data.get("source_event_ids", []),
                confidence=data.get("confidence", 0.8),
            )
        
        elif event_type == "CONTEXT":
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
    
    except Exception as e:
        print(f"Error parsing {event_type}: {e}")
        return None
    
    return None


def _parse_event_line(line: str, base_time: datetime, actors: Dict[str, Actor], parse_func) -> Optional[Any]:
    """Parse a single event line from LLM output."""
    if not line or line.startswith("#"):
        return None
    
    if "|" not in line:
        return None
    
    parts = line.split("|", 1)
    if len(parts) != 2:
        return None
    
    event_type, json_str = parts
    event_type = event_type.strip().upper()
    
    try:
        data = json.loads(json_str)
        return parse_func(event_type, data, base_time, actors)
    except json.JSONDecodeError:
        return None


# =============================================================================
# EVENT GENERATOR
# =============================================================================

class EventGenerator:
    """
    LLM-based event generator with streaming output.
    
    Pipeline:
        1. Scenario → LLM → Stream of Layer 1 events
        2. Layer 1 events → LLM → Stream of Layer 2 events
    
    Uses LLMClient for unified access to multiple providers.
    """
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize event generator.
        
        Args:
            model: Model identifier. If None, uses config default.
        """
        self.client = LLMClient(model=model)
        self.config = get_config()
        self.model = self.client.model
        self.actors: Dict[str, Actor] = {}
    
    def generate_layer1_events(
        self,
        scenario: Scenario,
        on_event: Optional[Callable[[Layer1Event], None]] = None,
    ) -> Generator[Layer1Event, None, None]:
        """
        Generate Layer 1 events from scenario description.
        Streams events when using Anthropic directly, batch mode for aisuite.
        """
        self.actors = {}
        base_time = datetime.now()
        
        # Build the prompt
        user_prompt = self._build_layer1_prompt(scenario)
        
        # Parse function for this layer
        def parse_line(line: str) -> Optional[Layer1Event]:
            return _parse_event_line(line, base_time, self.actors, parse_layer1_event)
        
        # Stream and parse events
        for event in self.client.stream_and_parse(
            LAYER1_SYSTEM_PROMPT,
            user_prompt,
            parse_line,
            max_tokens=self.config.layer1_max_tokens,
            on_parsed=on_event,
        ):
            yield event
    
    def generate_layer2_events(
        self,
        layer1_events: List[Layer1Event],
        scenario: Scenario,
        on_event: Optional[Callable[[Layer2Event], None]] = None,
    ) -> Generator[Layer2Event, None, None]:
        """
        Generate Layer 2 events from Layer 1 events.
        Streams events when using Anthropic directly, batch mode for aisuite.
        """
        base_time = layer1_events[0].timestamp if layer1_events else datetime.now()
        
        # Build the prompt
        user_prompt = self._build_layer2_prompt(layer1_events, scenario, base_time)
        
        # Parse function for this layer
        def parse_line(line: str) -> Optional[Layer2Event]:
            return _parse_event_line(line, base_time, self.actors, parse_layer2_event)
        
        # Stream and parse events
        for event in self.client.stream_and_parse(
            LAYER2_SYSTEM_PROMPT,
            user_prompt,
            parse_line,
            max_tokens=self.config.layer2_max_tokens,
            on_parsed=on_event,
        ):
            yield event
    
    def _build_layer1_prompt(self, scenario: Scenario) -> str:
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
    
    def _build_layer2_prompt(
        self, 
        layer1_events: List[Layer1Event], 
        scenario: Scenario, 
        base_time: datetime
    ) -> str:
        """Build the user prompt for Layer 2 generation."""
        l1_summary = self._format_layer1_for_prompt(layer1_events, base_time)
        
        return f"""Analyze these Layer 1 sensor events and generate Layer 2 interpreted events.

SCENARIO CONTEXT: {scenario.name}
{scenario.description}

LAYER 1 EVENTS:
{l1_summary}

Generate Layer 2 events (BEHAVIORAL, INTERACTION, CONTEXT) that capture the meaningful patterns:"""
    
    def _format_layer1_for_prompt(self, events: List[Layer1Event], base_time: datetime) -> str:
        """Format Layer 1 events for inclusion in Layer 2 prompt."""
        lines = []
        
        for event in events:
            offset = (event.timestamp - base_time).total_seconds()
            
            if isinstance(event, SpeechEvent):
                lines.append(
                    f"[{offset:.0f}s] {event.event_id} SPEECH: {event.speaker.id} said "
                    f"'{event.transcription or '[vocalization]'}' "
                    f"(target={event.target.value}, type={event.vocal_type.value})"
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
    
    def generate_session(
        self,
        scenario: Scenario,
        on_layer1_event: Optional[Callable[[Layer1Event], None]] = None,
        on_layer2_event: Optional[Callable[[Layer2Event], None]] = None,
        on_phase_change: Optional[Callable[[str], None]] = None,
    ) -> Session:
        """
        Generate a complete session with both event layers.
        
        Args:
            scenario: The scenario to generate
            on_layer1_event: Callback for each Layer 1 event (for streaming display)
            on_layer2_event: Callback for each Layer 2 event (for streaming display)
            on_phase_change: Callback when switching phases ("layer1" or "layer2")
        
        Returns:
            Complete Session object
        """
        if on_phase_change:
            on_phase_change("layer1")
        
        # Generate Layer 1
        layer1_events = list(self.generate_layer1_events(scenario, on_event=on_layer1_event))
        
        if on_phase_change:
            on_phase_change("layer2")
        
        # Generate Layer 2
        layer2_events = list(self.generate_layer2_events(layer1_events, scenario, on_event=on_layer2_event))
        
        # Build session
        session = Session(
            metadata=SessionMetadata(
                session_id=uuid.uuid4().hex[:12],
                start_time=layer1_events[0].timestamp if layer1_events else datetime.now(),
                end_time=layer1_events[-1].timestamp if layer1_events else datetime.now(),
                scenario_name=scenario.name,
                scenario_description=scenario.description,
                num_children=scenario.num_children,
                num_adults=scenario.num_adults,
                layer2_llm_model=self.model,
            ),
            actors=list(self.actors.values()),
        )
        
        # Sort events into appropriate lists
        for event in layer1_events:
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
        
        for event in layer2_events:
            if isinstance(event, BehavioralEvent):
                session.behavioral_events.append(event)
            elif isinstance(event, InteractionEvent):
                session.interaction_events.append(event)
            elif isinstance(event, ContextEvent):
                session.context_events.append(event)
        
        return session


# =============================================================================
# SCENARIO LOADING
# =============================================================================

def load_scenario(path: str) -> Scenario:
    """Load a scenario from a JSON file."""
    return Scenario.from_json(path)


def load_scenarios_from_directory(directory: str) -> Dict[str, Scenario]:
    """Load all scenarios from a directory."""
    import os
    scenarios = {}
    
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            path = os.path.join(directory, filename)
            try:
                scenario = load_scenario(path)
                scenarios[scenario.name] = scenario
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return scenarios
