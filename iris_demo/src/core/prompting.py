"""
Prompt templates and parsing helpers shared by generator and engine components.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from .schema import (
    Actor,
    ActorRole,
    ActivityType,
    AmbientAudioEvent,
    BehaviorCategory,
    BehavioralEvent,
    BodyOrientation,
    ClassroomClimate,
    ClassroomZone,
    CommunicativeIntent,
    ContentType,
    ContextEvent,
    EmotionalState,
    GazeDirection,
    GazeEvent,
    InteractionEvent,
    InteractionQuality,
    InteractionType,
    Intensity,
    Layer1Event,
    Layer2Event,
    Location,
    MovementType,
    ObjectAction,
    ObjectEvent,
    PostureEvent,
    PostureType,
    ProsodyFeatures,
    ProximityChange,
    ProximityEvent,
    ProximityLevel,
    SpeechEvent,
    SpeechTarget,
    TriggerType,
    VerbalComplexity,
    VocalType,
)


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
Focus on observable behaviors, not interpretations. Start with events that capture the context."""


LAYER2_SYSTEM_PROMPT = """You convert Layer 1 sensor traces into Layer 2 semantic events for an autism support classroom.

Use the schema precisely:
- BehavioralEvent → choose `category` from BehaviorCategory (stimming, task_engagement, distress, etc.), set `intensity` (low/moderate/high), `apparent_emotion` (calm, anxious, angry, ...), and `trigger` (social_demand, transition, peer_action, etc.). Provide a short `description`, optional `trigger_description`, note whether a regulation strategy was effective, and include `source_event_ids` referencing the raw sensor IDs you relied on.
- InteractionEvent → specify `interaction_type` (initiation, joint_attention, conflict, ...), `quality` (successful/partial/unsuccessful/ongoing), `content_type` (request, protest, greeting, ...), and `communicative_intent` (social_connection, self_regulation, ...). Make `description` summarize the social exchange, capture reciprocity or adult facilitation when relevant, and cite the source events.
- ContextEvent → summarize classroom conditions with `activity_type` (circle_time, free_play, transition, ...), `primary_zone`, `classroom_climate` (calm, energetic, tense, ...), and `noise_level`. Note transitions (is_transition, transition_from/to) plus adult presence.

Reasoning expectations:
1. Infer meaning from patterns that span multiple Layer 1 events before declaring a Layer 2 inference.
2. Keep a steady cadence of ContextEvents (roughly every 30-60 seconds of scenario time) so classroom climate stays updated.
3. Always populate `source_event_ids` with the IDs of the Layer 1 events that justify each inference; highlight confidence only when deviations from the default 0.8 are meaningful.
4. Remain clinically useful: emphasize behaviors that matter for intervention planning, regulation, participation, or peer/adult engagement.
5. When signals conflict or evidence is weak, use `CommunicativeIntent.UNCLEAR`, `TriggerType.UNKNOWN`, or similar neutral enums instead of speculating.

CRITICAL PRIVACY GUARDRAILS:
- NEVER include direct speech quotes (with quotation marks or apostrophes) in any description field
- NEVER use verbs like "said", "stated", "reported", "uttered", "spoke", "told" that imply direct speech capture
- Instead, describe speech using behavioral metadata: word_count, vocal_quality (loud/soft/calm/excited), communication target, prosody characteristics
- Example: Instead of 'Child said "I don't want to play"', write: 'Child vocalized protest with elevated volume and negative prosody, directed at peer'
- All descriptions must be clinically descriptive without capturing or referencing actual speech content
"""


RECONSTRUCT_SCENARIO_SYSTEM_PROMPT = """You are an expert at analyzing classroom observation data to reconstruct the key events and dynamics of a scenario.

Your task: Given a stream of Layer 2 (behavioral/social) inferred events, reconstruct a natural language description of what happened in the classroom during the observation.

Focus on:
1. Key behavioral episodes and their triggers
2. Important interactions and social moments
3. Overall classroom climate and transitions
4. Notable patterns in individual children's behavior
5. Adult support and facilitation

OUTPUT FORMAT:
Generate a cohesive narrative that captures the essential story of the observation session.
Write in professional clinical language suitable for an observation report.
Do NOT include any PII or identifying information.
Reference specific times (timestamp_offset_sec) to ground descriptions.
"""

SOCIAL_STORY_SYSTEM_PROMPT = """You are creating a social story for a child with autism based on a challenging situation observed in the classroom.

Based on the event data, identify the most significant challenging moment and create a surrogate social story that captures the essence of the challenges faced by the child, while changing all identifying details (names, settings, objects) to protect privacy:

1. **Title**: Simple, descriptive title
2. **Situation**: Describe what happened in simple, concrete terms (2-3 sentences). Use different genders and names than the actual child.
3. **Context:**: Choose a different setting than the original (e.g., park instead of classroom) (1-2 sentences). Also change the activity and any objects involved.
4. **Feeling**: Acknowledge the emotion the child might have felt (1-2 sentences)
5. **Strategy**: Provide 1-2 simple coping strategies (2-3 sentences)
6. **Positive Outcome**: Describe what happens when the strategy is used (1-2 sentences)
7. **Practice Phrase**: A simple phrase the child can remember

Use:
- First person perspective ("When I...")
- Simple, concrete language
- Present tense
- Short sentences
- Positive framing

Keep the total length to about 150-200 words.
Format with clear section headers."""

CLINICAL_REPORT_SYSTEM_PROMPT = """You are generating a clinical report for a Speech-Language Pathologist reviewing a classroom observation session.

Based on the event data provided, create a structured clinical report that includes:

1. **Session Overview**: Duration, setting, participants
2. **Communication Profile** (for each focus child):
   - Expressive language: complexity, frequency, targets
   - Receptive indicators: response to verbal input
   - Pragmatic skills: turn-taking, topic maintenance, social communication
3. **Social Interaction Patterns**:
   - Peer interactions: initiation vs response, quality, reciprocity
   - Adult interactions: help-seeking, compliance, joint attention
4. **Behavioral Observations**:
   - Self-regulation strategies observed and their effectiveness
   - Sensory responses and triggers
   - Emotional patterns
5. **Strengths Identified**: What's working well
6. **Areas for Support**: Specific intervention targets
7. **Recommendations**: 2-3 actionable next steps

Use clinical terminology appropriate for SLP documentation. Be specific with examples from the events.
"""


# =============================================================================
# PARSING HELPERS
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
    actors: Dict[str, Actor],
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

        if event_type == "AMBIENT":
            return AmbientAudioEvent(
                event_id=event_id,
                timestamp=timestamp,
                sound_type=data.get("sound_type", "unknown"),
                intensity=_safe_enum(Intensity, data.get("intensity"), Intensity.MODERATE),
                duration_ms=data.get("duration_ms", 0),
                location_estimate=_safe_enum(ClassroomZone, data.get("zone")),
            )

        if event_type == "PROXIMITY":
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

        if event_type == "GAZE":
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

        if event_type == "POSTURE":
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

        if event_type == "OBJECT":
            return ObjectEvent(
                event_id=event_id,
                timestamp=timestamp,
                actor=get_actor(data["actor_id"]),
                object_type=data.get("object_type", "unknown"),
                action=_safe_enum(ObjectAction, data.get("action"), ObjectAction.MANIPULATE),
                is_appropriate=data.get("is_appropriate"),
                shared_with=get_actor(data["shared_with_id"]) if data.get("shared_with_id") else None,
            )
    except Exception as exc:  # pragma: no cover - defensive parse
        print(f"Error parsing {event_type}: {exc}")
        return None

    return None


def parse_layer2_event(
    event_type: str,
    data: dict,
    base_time: datetime,
    actors: Dict[str, Actor],
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

        if event_type == "INTERACTION":
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

        if event_type == "CONTEXT":
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
    except Exception as exc:  # pragma: no cover - defensive parse
        print(f"Error parsing {event_type}: {exc}")
        return None

    return None


def parse_event_line(
    line: str,
    base_time: datetime,
    actors: Dict[str, Actor],
    parse_func,
) -> Optional[Any]:
    
    #print(f"Parsing line: {line}")
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
