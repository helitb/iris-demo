"""
IRIS Dual-Purpose App

1. Interactive Demo:
   - Describe or load a scenario
   - Choose English/Hebrew narrative language
   - Stream Layer 1 and Layer 2 events
   - View reconstructed narrative + SLP clinical + social story

2. Evaluation Dashboard:
   - Browse saved sessions by metadata
   - Load L1/L2 artifacts from persistence
   - Run per-stage evaluations with different LLMs
   - Compare two runs side-by-side
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Ensure local src package is importable
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (  # type: ignore
    L1EventLog,
    ObservationEngine,
    Scenario,
    Session,
    SessionMetadata,
    SpeechEvent,
    AmbientAudioEvent,
    ProximityEvent,
    GazeEvent,
    PostureEvent,
    ObjectEvent,
    BehavioralEvent,
    InteractionEvent,
    ContextEvent,
    get_available_models,
    get_config,
    list_saved_sessions,
    load_l1_log_from_session,
    load_scenarios_from_directory,
    serialize_session,
)

from src.core.session_store import SessionStore
from src.core.paths import resolve_sessions_directory
from src.core.storage import deserialize_event
from src.core.context import ScenarioContext
from src.core.session import Layer1Batch, Layer2Batch, ReconstructionArtifact, SessionHandle


# -----------------------------------------------------------------------------
# Streamlit page configuration + styles
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="IRIS Scenario Lab",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.event-card {
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 8px;
    font-size: 0.85em;
    border-left: 5px solid;
    font-family: 'Segoe UI', sans-serif;
}
.event-speech { background: #e3f2fd; border-color: #1976d2; }
.event-ambient { background: #e1f5fe; border-color: #0288d1; }
.event-proximity { background: #f3e5f5; border-color: #7b1fa2; }
.event-gaze { background: #e0f7fa; border-color: #00838f; }
.event-posture { background: #fff3e0; border-color: #ef6c00; }
.event-object { background: #fff8e1; border-color: #f9a825; }
.event-behavioral { background: #fce4ec; border-color: #c2185b; }
.event-interaction { background: #e8f5e9; border-color: #388e3c; }
.event-context { background: #ede7f6; border-color: #512da8; }
.phase-header {
    padding: 8px 16px;
    margin: 16px 0 8px 0;
    border-radius: 4px;
    font-weight: bold;
    font-size: 0.9em;
}
.phase-layer1 { background: #e3f2fd; color: #1565c0; }
.phase-layer2 { background: #fce4ec; color: #c2185b; }
.report-container {
    background: #f7f9fc;
    border-radius: 6px;
    padding: 16px;
    border: 1px solid #dce3f0;
    font-size: 0.92em;
}
.form-label {
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 4px;
    display: inline-block;
}
.stTabs [data-baseweb="tab-list"] button div p {
    font-size: 1.05rem;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Scenario helper
# -----------------------------------------------------------------------------

def save_scenario_to_disk(scenario: Scenario, directory: Path) -> str:
    directory.mkdir(exist_ok=True, parents=True)
    base = "".join(c if c.isalnum() else "_" for c in scenario.name).strip() or "scenario"
    filename = f"{base}.json"
    filepath = directory / filename
    counter = 1
    while filepath.exists():
        filename = f"{base}_{counter}.json"
        filepath = directory / filename
        counter += 1
    payload = {
        "name": scenario.name,
        "description": scenario.description,
        "duration_minutes": scenario.duration_minutes,
        "num_children": scenario.num_children,
        "num_adults": scenario.num_adults,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return filename


def load_l1_event_log_file(path: Path | str) -> L1EventLog:
    """Load an L1 event log JSON previously saved via L1EventLog.to_dict()."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    timestamp = data.get("timestamp_created")
    try:
        base_time = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
    except Exception:
        base_time = datetime.now()
    return L1EventLog.from_dict(data, base_time=base_time)


def _parse_iso_datetime(value: Optional[str], fallback: Optional[datetime] = None) -> datetime:
    if not value:
        return fallback or datetime.now()
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return fallback or datetime.now()


def _load_events_with_payload(path: Path) -> tuple[List, Dict[str, Any]]:
    if not path or not path.exists():
        return [], {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [], {}
    events: List = []
    for event_data in payload.get("events", []):
        try:
            events.append(deserialize_event(event_data))
        except Exception:
            continue
    return events, payload


def _build_session_from_events(metadata: SessionMetadata, l1_events: List, l2_events: List) -> Session:
    session = Session(metadata=metadata)
    actors: Dict[str, Any] = {}

    def collect(actor):
        if actor and getattr(actor, "id", None) and actor.id not in actors:
            actors[actor.id] = actor

    for event in l1_events:
        collect(getattr(event, "speaker", None))
        collect(getattr(event, "actor", None))
        collect(getattr(event, "target", None))
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

    for event in l2_events:
        collect(getattr(event, "actor", None))
        collect(getattr(event, "initiator", None))
        collect(getattr(event, "recipient", None))
        if isinstance(event, BehavioralEvent):
            session.behavioral_events.append(event)
        elif isinstance(event, InteractionEvent):
            session.interaction_events.append(event)
        elif isinstance(event, ContextEvent):
            session.context_events.append(event)

    session.actors = list(actors.values())
    return session


def _scenario_from_payload(payload: Dict[str, Any]) -> Scenario:
    return Scenario(
        name=payload.get("name") or "Unnamed scenario",
        description=payload.get("description") or "",
        duration_minutes=payload.get("duration_minutes") or 1,
        num_children=payload.get("num_children") or 0,
        num_adults=payload.get("num_adults") or 0,
        focus_children=payload.get("focus_children"),
        key_moments=payload.get("key_moments"),
    )


def _build_layer1_info(events: List, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not events:
        return None
    started = _parse_iso_datetime(payload.get("started_at"))
    ended = _parse_iso_datetime(payload.get("ended_at"), fallback=started)
    return {"events": events, "started_at": started, "ended_at": ended}


def _build_layer2_info(events: List, payload: Dict[str, Any], sanitized: bool) -> Optional[Dict[str, Any]]:
    if not events:
        return None
    generated = _parse_iso_datetime(payload.get("generated_at"))
    return {"events": events, "generated_at": generated, "sanitized": sanitized}


def _load_session_components(session_dir: Path) -> Optional[Dict[str, Any]]:
    scenario_path = session_dir / "scenario.json"
    metadata_path = session_dir / "metadata.json"
    if not scenario_path.exists() or not metadata_path.exists():
        return None
    try:
        scenario_payload = json.loads(scenario_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    scenario = _scenario_from_payload(scenario_payload)
    l1_events, l1_payload = _load_events_with_payload(session_dir / "layer1_raw.json")
    l2_raw_events, l2_raw_payload = _load_events_with_payload(session_dir / "layer2_raw.json")
    l2_sanitized_events, l2_sanitized_payload = _load_events_with_payload(
        session_dir / "layer2_sanitized.json"
    )

    l1_info = _build_layer1_info(l1_events, l1_payload) if l1_events else None
    l2_raw_info = _build_layer2_info(l2_raw_events, l2_raw_payload, sanitized=False) if l2_raw_events else None
    l2_sanitized_info = (
        _build_layer2_info(l2_sanitized_events, l2_sanitized_payload, sanitized=True)
        if l2_sanitized_events
        else None
    )
    if not l2_sanitized_info:
        l2_sanitized_info = l2_raw_info

    return {
        "scenario": scenario,
        "metadata": metadata,
        "language": metadata.get("language", "en"),
        "models": metadata.get("models", {}) or {},
        "l1": l1_info,
        "l2_raw": l2_raw_info,
        "l2_sanitized": l2_sanitized_info,
        "reconstruction": _read_report_file(session_dir / "reconstruction.txt"),
        "slp": _read_report_file(session_dir / "slp_report.txt"),
        "social": _read_report_file(session_dir / "social_story.txt"),
        "source_session_id": metadata.get("session_id"),
    }


def _create_handle_from_components(
    components: Dict[str, Any],
    include_reports: bool = False,
    language_override: Optional[str] = None,
) -> SessionHandle:
    scenario = components["scenario"]
    context = ScenarioContext(scenario=scenario, source="replay")
    handle = SessionHandle(session_id=uuid.uuid4().hex[:12], context=context)
    language = language_override or components.get("language") or "en"
    setattr(handle, "language", language)

    for stage, model_id in components.get("models", {}).items():
        handle.set_model(stage, model_id)

    l1_info = components.get("l1")
    if l1_info:
        handle.attach_layer1(
            Layer1Batch(
                events=l1_info["events"],
                started_at=l1_info["started_at"],
                ended_at=l1_info["ended_at"],
            )
        )

    l2_raw_info = components.get("l2_raw")
    if l2_raw_info:
        handle.attach_layer2_raw(
            Layer2Batch(
                events=l2_raw_info["events"],
                generated_at=l2_raw_info["generated_at"],
                sanitized=False,
            )
        )

    l2_sanitized_info = components.get("l2_sanitized")
    if l2_sanitized_info:
        handle.attach_layer2_sanitized(
            Layer2Batch(
                events=l2_sanitized_info["events"],
                generated_at=l2_sanitized_info["generated_at"],
                sanitized=l2_sanitized_info.get("sanitized", True),
            )
        )

    if include_reports:
        text = components.get("reconstruction")
        if text:
            handle.attach_reconstruction(ReconstructionArtifact(text=text))
        slp_text = components.get("slp")
        if slp_text:
            handle.attach_slp_report(ReconstructionArtifact(text=slp_text, type="slp_report"))
        social_text = components.get("social")
        if social_text:
            handle.attach_social_story(ReconstructionArtifact(text=social_text, type="social_story"))

    return handle


def _components_from_scenario(scenario: Scenario) -> Dict[str, Any]:
    return {
        "scenario": scenario,
        "metadata": {},
        "language": "en",
        "models": {},
        "l1": None,
        "l2_raw": None,
        "l2_sanitized": None,
        "reconstruction": None,
        "slp": None,
        "social": None,
        "source_session_id": None,
    }


def _components_from_l1_log(l1_log: L1EventLog) -> Dict[str, Any]:
    scenario = Scenario(
        name=getattr(l1_log, "scenario_name", None) or "L1 log session",
        description=getattr(l1_log, "scenario_description", None) or "",
        duration_minutes=max(
            1,
            (getattr(l1_log, "duration_seconds", None) or 60) // 60 or 1,
        ),
        num_children=getattr(l1_log, "num_children", None) or 0,
        num_adults=getattr(l1_log, "num_adults", None) or 0,
    )
    events = list(l1_log.l1_events)
    if events:
        started = events[0].timestamp
        ended = events[-1].timestamp
    else:
        started = ended = datetime.now()
    l1_info = {"events": events, "started_at": started, "ended_at": ended}
    models = {}
    if l1_log.llm_model:
        models["layer1_llm_model"] = l1_log.llm_model
    return {
        "scenario": scenario,
        "metadata": {"session_id": getattr(l1_log, "log_id", None)},
        "language": getattr(l1_log, "language", None) or "en",
        "models": models,
        "l1": l1_info,
        "l2_raw": None,
        "l2_sanitized": None,
        "reconstruction": None,
        "slp": None,
        "social": None,
        "source_session_id": getattr(l1_log, "log_id", None),
    }


def _run_from_components(
    stage: str,
    components: Dict[str, Any],
    models: Dict[str, Optional[str]],
    language: str,
    sessions_dir: Path,
) -> Dict[str, Any]:
    handle = _create_handle_from_components(components, include_reports=False, language_override=language)
    store = SessionStore(str(sessions_dir))
    store.persist(handle)
    initial_model = models.get("layer1") or models.get("layer2") or models.get("reconstruction")
    engine = ObservationEngine(model_id=initial_model, artifact_lan=language, store=store)

    def swap_model(target: Optional[str]):
        if not target:
            return
        if engine.llm_client.model != target:
            engine.swap_model(target)

    if stage == "scenario":
        if not models.get("layer1") or not models.get("layer2"):
            raise ValueError("Configure Layer 1 and Layer 2 models to run from a scenario.")
        swap_model(models["layer1"])
        engine.run_layer1(handle)
        swap_model(models["layer2"])
        engine.run_layer2(handle)
        engine.sanitize_layer2(handle)
    elif stage == "l1":
        if not handle.artifacts.layer1_raw:
            raise ValueError("Layer 1 artifacts missing from source.")
        if not models.get("layer2"):
            raise ValueError("Select a Layer 2 model to continue from Layer 1 artifacts.")
        swap_model(models["layer2"])
        engine.run_layer2(handle)
        engine.sanitize_layer2(handle)
    elif stage == "l2_raw":
        if not handle.artifacts.layer2_raw:
            raise ValueError("Layer 2 raw artifacts missing from source.")
        engine.sanitize_layer2(handle)
    elif stage == "l2_sanitized":
        if not handle.artifacts.layer2_sanitized:
            raise ValueError("Layer 2 sanitized artifacts missing from source.")
    else:
        raise ValueError(f"Unsupported stage '{stage}'.")

    recon_model = models.get("reconstruction")
    if not recon_model:
        raise ValueError("Select a reconstruction model to finish the pipeline.")
    swap_model(recon_model)
    artifact = engine.reconstruct(handle)
    session = build_session_from_handle(handle)
    return {
        "handle": handle,
        "session": session,
        "language": language,
        "narrative": artifact.text,
        "models": {
            key: value
            for key, value in models.items()
            if value
        },
        "scenario_name": components["scenario"].name,
    }


def load_saved_session(session_dir: Path) -> Optional[Dict[str, Any]]:
    components = _load_session_components(session_dir)
    if not components:
        return None

    metadata_doc = components.get("metadata", {})
    models = components.get("models", {})
    scenario = components["scenario"]
    l1_info = components.get("l1")
    l2_info = components.get("l2_sanitized") or components.get("l2_raw")

    start_time = l1_info["started_at"] if l1_info else _parse_iso_datetime(metadata_doc.get("created_at"))
    end_time = l1_info["ended_at"] if l1_info else start_time

    metadata = SessionMetadata(
        session_id=metadata_doc.get("session_id") or session_dir.name,
        start_time=start_time,
        end_time=end_time,
        scenario_name=scenario.name,
        scenario_description=scenario.description,
        num_children=scenario.num_children,
        num_adults=scenario.num_adults,
        layer1_llm_model=models.get("layer1_llm_model"),
        layer2_llm_model=models.get("layer2_llm_model"),
        reconstruction_llm_model=models.get("reconstruction_llm_model"),
        language=components.get("language", "en"),
    )

    l1_events = l1_info["events"] if l1_info else []
    l2_events = l2_info["events"] if l2_info else []
    session = _build_session_from_events(metadata, l1_events, l2_events)
    narrative = components.get("reconstruction")
    models_summary = {
        "layer1": metadata.layer1_llm_model,
        "layer2": metadata.layer2_llm_model,
        "reconstruction": metadata.reconstruction_llm_model,
    }

    return {
        "session": session,
        "narrative": narrative or "Reconstruction not found for this session.",
        "language": metadata.language,
        "models": models_summary,
        "scenario_name": metadata.scenario_name or session_dir.name,
    }


# -----------------------------------------------------------------------------
# Session state helpers
# -----------------------------------------------------------------------------

def _default_demo_state() -> Dict:
    return {
        "scenario": None,
        "language": "en",
        "layer1_events": [],
        "layer2_events": [],
        "narrative": None,
        "reports": {},
        "session": None,
        "saved_session_path": None,
    }


def _init_session_state() -> None:
    if "demo_result" not in st.session_state:
        st.session_state.demo_result = _default_demo_state()
    if "eval_runs" not in st.session_state:
        st.session_state.eval_runs = {"A": None, "B": None}
    if "demo_running" not in st.session_state:
        st.session_state.demo_running = False


# -----------------------------------------------------------------------------
# Event formatting utilities
# -----------------------------------------------------------------------------

def format_timestamp(event) -> str:
    if not getattr(event, "timestamp", None):
        return "--:--"
    return event.timestamp.strftime("%H:%M:%S")


def format_actor(actor) -> str:
    if not actor:
        return "unknown"
    icon = "🧒" if actor.role.value == "child" else "🧑‍🏫"
    return f"{icon} {actor.id}"


def format_layer1_event(event) -> str:
    ts = format_timestamp(event)
    if isinstance(event, SpeechEvent):
        text = event.transcription or "[vocalization]"
        return f"""
        <div class="event-card event-speech">
            <span class="timestamp">[{ts}]</span> 🗣️ {format_actor(event.speaker)}
            <br><b>{text}</b>
            <br><small>{event.vocal_type.value} → {event.target.value} | words: {event.word_count}</small>
        </div>
        """
    if isinstance(event, AmbientAudioEvent):
        return f"""
        <div class="event-card event-ambient">
            <span class="timestamp">[{ts}]</span> 🔊 {event.sound_type}
            <br><small>intensity: {event.intensity.value} | duration: {event.duration_ms}ms</small>
        </div>
        """
    if isinstance(event, ProximityEvent):
        target = format_actor(event.target) if event.target else "unknown"
        return f"""
        <div class="event-card event-proximity">
            <span class="timestamp">[{ts}]</span> 📏 {format_actor(event.actor)} {event.change_type.value} {target}
            <br><small>level: {event.proximity_level.value}</small>
        </div>
        """
    if isinstance(event, GazeEvent):
        target = (
            format_actor(event.target_actor)
            if event.target_actor
            else event.target_object
            or "zone"
        )
        mutual = " 👀 mutual" if event.is_mutual else ""
        return f"""
        <div class="event-card event-gaze">
            <span class="timestamp">[{ts}]</span> 👁️ {format_actor(event.actor)} &rarr; {target}{mutual}
        </div>
        """
    if isinstance(event, PostureEvent):
        rep = " 🔁 repetitive" if event.is_repetitive else ""
        return f"""
        <div class="event-card event-posture">
            <span class="timestamp">[{ts}]</span> 🧍 {format_actor(event.actor)} {event.posture.value}
            <br><small>{event.movement.value}{rep}</small>
        </div>
        """
    if isinstance(event, ObjectEvent):
        shared = f" with {format_actor(event.shared_with)}" if event.shared_with else ""
        return f"""
        <div class="event-card event-object">
            <span class="timestamp">[{ts}]</span> 🧩 {format_actor(event.actor)} {event.action.value} <b>{event.object_type}</b>{shared}
        </div>
        """
    return f"<div class='event-card'>[{ts}] Unknown Layer 1 event</div>"


def format_layer2_event(event) -> str:
    ts = format_timestamp(event)
    if isinstance(event, BehavioralEvent):
        return f"""
        <div class="event-card event-behavioral">
            <span class="timestamp">[{ts}]</span> 🧠 {format_actor(event.actor)} · <b>{event.category.value}</b>
            <br>{event.description}
            <br><small>emotion: {event.apparent_emotion.value} · intensity: {event.intensity.value}</small>
        </div>
        """
    if isinstance(event, InteractionEvent):
        return f"""
        <div class="event-card event-interaction">
            <span class="timestamp">[{ts}]</span> 🤝 {format_actor(event.initiator)} → {format_actor(event.recipient)}
            <br><b>{event.interaction_type.value}</b>: {event.description}
            <br><small>quality: {event.quality.value}</small>
        </div>
        """
    if isinstance(event, ContextEvent):
        transition = " 🔀 transition" if event.is_transition else ""
        return f"""
        <div class="event-card event-context">
            <span class="timestamp">[{ts}]</span> 🏫 {event.activity_type.value} @ {event.primary_zone.value}{transition}
            <br><small>climate: {event.classroom_climate.value} · noise: {event.noise_level.value}</small>
        </div>
        """
    return f"<div class='event-card'>[{ts}] Unknown Layer 2 event</div>"


# -----------------------------------------------------------------------------
# Event filtering helpers
# -----------------------------------------------------------------------------

def filter_layer1_events(events: List, selected: List[str]) -> List:
    if not selected or "All" in selected:
        return events
    type_map = {
        "Speech": SpeechEvent,
        "Ambient": AmbientAudioEvent,
        "Proximity": ProximityEvent,
        "Gaze": GazeEvent,
        "Posture": PostureEvent,
        "Object": ObjectEvent,
    }
    allowed = tuple(type_map[name] for name in selected if name in type_map)
    return [evt for evt in events if isinstance(evt, allowed)]


def filter_layer2_events(events: List, selected: List[str]) -> List:
    if not selected or "All" in selected:
        return events
    type_map = {
        "Behavioral": BehavioralEvent,
        "Interaction": InteractionEvent,
        "Context": ContextEvent,
    }
    allowed = tuple(type_map[name] for name in selected if name in type_map)
    return [evt for evt in events if isinstance(evt, allowed)]


# -----------------------------------------------------------------------------
# Session builder
# -----------------------------------------------------------------------------

def build_session_from_handle(handle) -> Optional[Session]:
    artifacts = handle.artifacts
    if not artifacts.layer1_raw:
        return None
    scenario = handle.context.scenario
    metadata = SessionMetadata(
        session_id=handle.session_id,
        start_time=artifacts.layer1_raw.started_at,
        end_time=artifacts.layer1_raw.ended_at,
        scenario_name=scenario.name,
        scenario_description=scenario.description,
        num_children=scenario.num_children,
        num_adults=scenario.num_adults,
        layer1_llm_model=handle.models.get("layer1_llm_model"),
        layer2_llm_model=handle.models.get("layer2_llm_model"),
        reconstruction_llm_model=handle.models.get("reconstruction_llm_model"),
        language=getattr(handle, "language", "en"),
    )
    session = Session(metadata=metadata)
    actors: Dict[str, object] = {}

    def collect(actor):
        if actor and getattr(actor, "id", None) and actor.id not in actors:
            actors[actor.id] = actor

    for event in artifacts.layer1_raw.events:
        collect(getattr(event, "speaker", None))
        collect(getattr(event, "actor", None))
        collect(getattr(event, "target", None))
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

    l2_events = []
    if artifacts.layer2_sanitized:
        l2_events = artifacts.layer2_sanitized.events
    elif artifacts.layer2_raw:
        l2_events = artifacts.layer2_raw.events

    for event in l2_events:
        collect(getattr(event, "actor", None))
        collect(getattr(event, "initiator", None))
        collect(getattr(event, "recipient", None))
        if isinstance(event, BehavioralEvent):
            session.behavioral_events.append(event)
        elif isinstance(event, InteractionEvent):
            session.interaction_events.append(event)
        elif isinstance(event, ContextEvent):
            session.context_events.append(event)

    session.actors = list(actors.values())
    return session


# -----------------------------------------------------------------------------
# Report helpers
# -----------------------------------------------------------------------------

def _read_report_file(path: Path) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or None
    except Exception:
        return None


def poll_for_engine_reports(session_dir: Path, timeout: float = 60.0, poll_interval: float = 1.0) -> Dict[str, str]:
    """Poll the session directory until engine-generated reports appear."""
    targets = {
        "slp_clinical": session_dir / "slp_report.txt",
        "social_story": session_dir / "social_story.txt",
    }
    reports: Dict[str, str] = {}
    deadline = time.time() + timeout
    while time.time() < deadline and len(reports) < len(targets):
        for key, path in targets.items():
            if key in reports or not path.exists():
                continue
            text = _read_report_file(path)
            if text:
                reports[key] = text
        if len(reports) < len(targets):
            time.sleep(poll_interval)
    return reports


# -----------------------------------------------------------------------------
# Demo execution
# -----------------------------------------------------------------------------

def run_demo_pipeline(
    scenario: Scenario,
    model_id: str,
    language: str,
    l1_placeholder,
    l2_placeholder,
    sessions_dir: Path,
) -> Dict:
    store = SessionStore(str(sessions_dir))
    engine = ObservationEngine(
        model_id=model_id,
        artifact_lan=language,
        store=store,
    )
    handle = engine.create_session(scenario)

    layer1_events: List = []
    layer2_events: List = []

    def on_l1(event):
        layer1_events.append(event)
        l1_placeholder.markdown(
            "".join(format_layer1_event(evt) for evt in layer1_events[-30:]),
            unsafe_allow_html=True,
        )

    def on_l2(event):
        layer2_events.append(event)
        l2_placeholder.markdown(
            "".join(format_layer2_event(evt) for evt in layer2_events[-20:]),
            unsafe_allow_html=True,
        )

    with st.spinner("Generating Layer 1 events..."):
        engine.run_layer1(handle, on_event=on_l1)

    with st.spinner("Inferring Layer 2 events..."):
        engine.run_layer2(handle, on_event=on_l2)
        engine.sanitize_layer2(handle)

    with st.spinner("Reconstructing narrative..."):
        artifact = engine.reconstruct(handle)

    session = build_session_from_handle(handle)
    session_dir = Path(store.base_directory) / handle.session_id
    with st.spinner("Waiting for reports..."):
        reports = poll_for_engine_reports(session_dir)
    
    return {
        "scenario": scenario,
        "language": language,
        "layer1_events": session.layer1_events if session else layer1_events,
        "layer2_events": session.layer2_events if session else layer2_events,
        "narrative": artifact.text,
        "reports": reports,
        "session": session,
    }


# -----------------------------------------------------------------------------
# Demo tab
# -----------------------------------------------------------------------------

def render_demo_tab(
    scenarios_dir: Path,
    sessions_dir: Path,
    available_models: Dict[str, str],
    config_model: str,
):
    st.subheader("🎬 Interactive Demo")
    scenario_options = load_scenarios_from_directory(str(scenarios_dir))
    st.markdown("<span class='form-label'>Scenario Source</span>", unsafe_allow_html=True)
    scenario_mode = st.radio(
        "Scenario Source",
        ["Saved scenario", "Custom description"],
        horizontal=True,
        label_visibility="collapsed",
    )
    scenario: Optional[Scenario] = None
    is_custom = scenario_mode == "Custom description"
    if scenario_mode == "Saved scenario":
        if scenario_options:
            names = list(scenario_options.keys())
            st.markdown("<span class='form-label'>Select scenario</span>", unsafe_allow_html=True)
            chosen = st.selectbox("Select scenario", names, label_visibility="collapsed")
            scenario = scenario_options[chosen]
            with st.expander("Scenario details", expanded=False):
                st.write(scenario.description)
                st.caption(f"{scenario.num_children} children · {scenario.num_adults} adults · {scenario.duration_minutes} min")
        else:
            st.info("No scenarios in ./scenarios yet.")
    else:
        name = st.text_input("Scenario name", "Custom Scenario")
        description = st.text_area("Describe what's happening", height=140)
        duration = st.slider("Duration (minutes)", 1, 10, 5)
        num_children = st.slider("Number of children", 1, 8, 3)
        num_adults = st.slider("Number of adults", 1, 4, 1)
        if description:
            scenario = Scenario(
                name=name,
                description=description,
                duration_minutes=duration,
                num_children=num_children,
                num_adults=num_adults,
            )
            if st.checkbox("Save scenario to disk"):
                if st.button("💾 Save scenario"):
                    filename = save_scenario_to_disk(scenario, scenarios_dir)
                    st.success(f"Saved as {filename}")
        else:
            st.info("Enter a description to enable the demo run.")

    model_names = list(available_models.keys())
    default_idx = 0
    for idx, name in enumerate(model_names):
        if available_models[name] == config_model:
            default_idx = idx
            break

    lang_col, model_col = st.columns(2)
    with lang_col:
        st.markdown("<span class='form-label'>Narrative language</span>", unsafe_allow_html=True)
        language_choice = st.radio(
            "Narrative language",
            ["English", "Hebrew"],
            horizontal=True,
            index=0,
            label_visibility="collapsed",
        )
        language = "he" if language_choice == "Hebrew" else "en"

    with model_col:
        st.markdown("<span class='form-label'>Pipeline LLM</span>", unsafe_allow_html=True)
        selected_model_name = st.selectbox(
            "Pipeline LLM",
            model_names,
            index=default_idx,
            label_visibility="collapsed",
        )
    selected_model = available_models[selected_model_name]

    l1_expander = st.expander("Layer 1 Stream (raw events)", expanded=True)
    with l1_expander:
        l1_filters = st.multiselect(
            "Filter Layer 1 types",
            ["All", "Speech", "Ambient", "Proximity", "Gaze", "Posture", "Object"],
            default=["All"],
            key="demo_l1_filters",
        )
        l1_placeholder = st.empty()

    l2_expander = st.expander("Layer 2 Stream (LLM inference)", expanded=True)
    with l2_expander:
        l2_filters = st.multiselect(
            "Filter Layer 2 types",
            ["All", "Behavioral", "Interaction", "Context"],
            default=["All"],
            key="demo_l2_filters",
        )
        l2_placeholder = st.empty()

    api_key_present = os.environ.get("ANTHROPIC_API_KEY")
    disabled = scenario is None or not selected_model or not api_key_present
    if not api_key_present:
        st.warning("ANTHROPIC_API_KEY missing – set it in your environment to run the demo.")
    run_button = st.empty()
    if st.session_state.demo_running:
        run_button.button(
            "⏳ Running…",
            disabled=True,
            use_container_width=True,
            key="run_demo_button",
        )
    else:
        if run_button.button(
            "🚀 Run Demo",
            disabled=disabled,
            use_container_width=True,
            key="run_demo_button",
        ):
            st.session_state.demo_running = True
            st.session_state.demo_result = _default_demo_state()
            l1_placeholder.empty()
            l2_placeholder.empty()
            try:
                result = run_demo_pipeline(
                    scenario=scenario,
                    model_id=selected_model,
                    language=language,
                    l1_placeholder=l1_placeholder,
                    l2_placeholder=l2_placeholder,
                    sessions_dir=sessions_dir,
                )
                st.session_state.demo_result = result
                st.success("Demo run complete.")
            finally:
                st.session_state.demo_running = False

    demo_result = st.session_state.demo_result
    if demo_result["layer1_events"]:
        filtered = filter_layer1_events(demo_result["layer1_events"], st.session_state.get("demo_l1_filters", ["All"]))
        l1_placeholder.markdown(
            "".join(format_layer1_event(evt) for evt in filtered),
            unsafe_allow_html=True,
        )
    if demo_result["layer2_events"]:
        filtered = filter_layer2_events(demo_result["layer2_events"], st.session_state.get("demo_l2_filters", ["All"]))
        l2_placeholder.markdown(
            "".join(format_layer2_event(evt) for evt in filtered),
            unsafe_allow_html=True,
        )

    if demo_result["narrative"]:
        st.markdown("### 🧾 Reconstructed Narrative")
        st.info(f"Language: {'English' if demo_result['language']=='en' else 'Hebrew'}")
        st.markdown(demo_result["narrative"])

        with st.expander("🩺 SLP Clinical Report", expanded=False):
            st.markdown(
                f"<div class='report-container'>{demo_result['reports'].get('slp_clinical','')}</div>",
                unsafe_allow_html=True,
            )

        with st.expander("🌱 Social Story", expanded=False):
            st.markdown(
                f"<div class='report-container'>{demo_result['reports'].get('social_story','')}</div>",
                unsafe_allow_html=True,
            )

        if demo_result["session"]:
            session_json = json.dumps(serialize_session(demo_result["session"]), ensure_ascii=False, indent=2)
            st.download_button(
                "⬇️ Download session JSON",
                session_json,
                file_name=f"iris_session_{demo_result['session'].metadata.session_id}.json",
                mime="application/json",
                use_container_width=True,
            )


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------

def list_engine_session_dirs(sessions_dir: Path) -> List[Path]:
    if not sessions_dir.exists():
        return []
    return [
        path
        for path in sessions_dir.iterdir()
        if path.is_dir() and (path / "layer1_raw.json").exists()
    ]


def _format_model_summary(models: Dict[str, Any]) -> str:
    if not models:
        return "LLM models unavailable"
    label_map = {
        "layer1_llm_model": "L1",
        "layer2_llm_model": "L2",
        "reconstruction_llm_model": "Recon",
    }
    parts: List[str] = []
    for key, label in label_map.items():
        value = models.get(key)
        if value:
            parts.append(f"{label}: {value}")
    for key, value in models.items():
        if key in label_map or not value:
            continue
        parts.append(f"{key}: {value}")
    return " | ".join(parts) if parts else "LLM models unavailable"


def _format_language_display(language: Optional[str]) -> str:
    if not language:
        return "Language: Unknown"
    name_map = {"en": "English", "he": "Hebrew"}
    label = name_map.get(language.lower(), language.upper())
    return f"Language: {label}"


def _load_session_summary(session_dir: Path) -> Optional[Dict[str, Any]]:
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    scenario_name = None
    scenario_path = session_dir / "scenario.json"
    if scenario_path.exists():
        try:
            scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
            scenario_name = scenario.get("name")
        except Exception:
            scenario_name = None
    created_at = meta.get("created_at")
    created_dt = None
    created_display = created_at or "Unknown timestamp"
    if created_at:
        try:
            created_dt = datetime.fromisoformat(created_at)
            created_display = created_dt.strftime("%b %d, %Y %H:%M")
        except Exception:
            created_dt = None
    models = meta.get("models", {})
    language = meta.get("language")
    return {
        "filename": session_dir.name,
        "path": str(session_dir),
        "session_id": meta.get("session_id") or session_dir.name,
        "scenario_name": scenario_name,
        "created_at": created_at,
        "created_dt": created_dt,
        "created_display": created_display,
        "language": language,
        "language_display": _format_language_display(language),
        "models": models,
        "models_display": _format_model_summary(models),
    }


def _render_file_section(label: str, path: Path, kind: str = "text") -> None:
    with st.expander(label, expanded=False):
        if not path.exists():
            st.info("File not found.")
            return
        try:
            if kind == "json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                pretty = json.dumps(payload, ensure_ascii=False, indent=2)
                st.code(pretty, language="json")
            else:
                text = path.read_text(encoding="utf-8").strip()
                if not text:
                    st.info("File is empty.")
                    return
                language = "markdown" if label.lower().startswith(("reconstructed", "slp", "social")) else "text"
                st.code(text, language=language)
        except Exception as exc:
            st.error(f"Unable to load {label.lower()}: {exc}")


def render_session_browser(sessions_dir: Path):
    if "session_browser_selected" not in st.session_state:
        st.session_state.session_browser_selected = None

    with st.expander("📁 Session Browser", expanded=False):
        saved_sessions = []
        for session_dir in list_engine_session_dirs(sessions_dir):
            summary = _load_session_summary(session_dir)
            if summary:
                saved_sessions.append(summary)
        if not saved_sessions:
            st.info("No saved sessions yet.")
            return
        saved_sessions.sort(key=lambda entry: entry.get("created_dt") or datetime.min, reverse=True)

        filter_query = st.text_input(
            "Filter by scenario or session id",
            key="session_browser_filter",
            placeholder="Type to filter saved sessions…",
        ).strip()
        filtered = [
            entry
            for entry in saved_sessions
            if not filter_query
            or filter_query.lower() in entry["session_id"].lower()
            or filter_query.lower() in (entry.get("scenario_name") or "").lower()
        ]
        if not filtered:
            st.warning("No sessions match your filter.")

        max_entries = 50
        for entry in filtered[:max_entries]:
            cols = st.columns([5, 1])
            with cols[0]:
                scenario_name = entry.get("scenario_name") or "Unknown scenario"
                st.markdown(
                    f"**Session {entry['session_id']} · {scenario_name}**"
                )
                st.caption(f"{entry['created_display']} · {entry['language_display']} · {entry['models_display']}")
                st.caption(f"Path: `{entry['path']}`")
            with cols[1]:
                if st.button("Load session", key=f"load_session_{entry['filename']}"):
                    st.session_state.session_browser_selected = entry["path"]
        selected_path = st.session_state.get("session_browser_selected")
        if selected_path:
            st.divider()
            session_dir = Path(selected_path)
            summary = _load_session_summary(session_dir)
            if summary:
                st.markdown(
                    f"#### Loaded session {summary['session_id']} · {summary.get('scenario_name') or 'Unknown scenario'}"
                )
                st.caption(f"{summary['created_display']} · {summary['language_display']} · {summary['models_display']}")
            else:
                st.markdown(f"#### Loaded session at {selected_path}")
            layer1_path = session_dir / "layer1_raw.json"
            layer2_path = session_dir / "layer2_sanitized.json"
            if not layer2_path.exists():
                layer2_path = session_dir / "layer2_raw.json"
            reconstruction_path = session_dir / "reconstruction.txt"
            slp_path = session_dir / "slp_report.txt"
            social_story_path = session_dir / "social_story.txt"

            _render_file_section("Layer 1 events", layer1_path, kind="json")
            _render_file_section("Layer 2 events", layer2_path, kind="json")
            _render_file_section("Reconstructed narrative", reconstruction_path)
            _render_file_section("SLP clinical report", slp_path)
            _render_file_section("Social story", social_story_path)


def execute_evaluation_run(
    slot_key: str,
    source_mode: str,
    scenario: Optional[Scenario],
    l1_source: Optional[Path],
    models: Dict[str, Optional[str]],
    language: Optional[str],
    sessions_dir: Path,
    l2_source: Optional[Path] = None,
) -> Optional[Dict]:
    try:
        language_choice = language or "en"

        if source_mode == "Scenario":
            if not scenario:
                st.error(f"[{slot_key}] Select a scenario first.")
                return None
            components = _components_from_scenario(scenario)
            return _run_from_components(
                stage="scenario",
                components=components,
                models=models,
                language=language_choice,
                sessions_dir=sessions_dir,
            )

        if source_mode == "Session L1 artifacts":
            if not l1_source:
                st.error(f"[{slot_key}] Select a Layer 1 source.")
                return None
            if l1_source.is_dir():
                components = _load_session_components(l1_source)
                if not components or not components.get("l1"):
                    st.error(f"[{slot_key}] Layer 1 artifacts missing in {l1_source}.")
                    return None
            else:
                l1_log = load_l1_event_log_file(str(l1_source))
                components = _components_from_l1_log(l1_log)
            return _run_from_components(
                stage="l1",
                components=components,
                models=models,
                language=language_choice,
                sessions_dir=sessions_dir,
            )

        if source_mode == "Session L2 artifacts":
            if not l2_source:
                st.error(f"[{slot_key}] Select a session containing Layer 2 events.")
                return None
            components = _load_session_components(l2_source)
            if not components or not components.get("l2_raw"):
                st.error(f"[{slot_key}] Layer 2 raw artifacts missing in {l2_source}.")
                return None
            return _run_from_components(
                stage="l2_raw",
                components=components,
                models=models,
                language=language_choice,
                sessions_dir=sessions_dir,
            )

        if source_mode == "Session L2 sanitized artifacts":
            if not l2_source:
                st.error(f"[{slot_key}] Select a session containing sanitized Layer 2 events.")
                return None
            components = _load_session_components(l2_source)
            if not components or not components.get("l2_sanitized"):
                st.error(f"[{slot_key}] Layer 2 sanitized artifacts missing in {l2_source}.")
                return None
            return _run_from_components(
                stage="l2_sanitized",
                components=components,
                models=models,
                language=language_choice,
                sessions_dir=sessions_dir,
            )

        st.error(f"[{slot_key}] Unsupported source mode {source_mode}.")
        return None
    except Exception as exc:
        st.error(f"[{slot_key}] Evaluation run failed: {exc}")
        return None


def render_eval_slot(
    slot_key: str,
    scenarios_dir: Path,
    sessions_dir: Path,
    available_models: Dict[str, str],
    default_model: str,
):
    container = st.container()
    container.markdown(f"#### Run {slot_key}")
    source_mode = container.radio(
        "Input source",
        ["Scenario", "Session L1 artifacts", "Session L2 artifacts", "Session L2 sanitized artifacts"],
        key=f"eval_source_{slot_key}",
    )
    scenario = None
    l1_source_path = None
    l2_source_path = None

    if source_mode == "Scenario":
        scenario_map = load_scenarios_from_directory(str(scenarios_dir))
        if scenario_map:
            scenario_name = container.selectbox(
                "Scenario",
                list(scenario_map.keys()),
                key=f"eval_scenario_{slot_key}",
            )
            scenario = scenario_map[scenario_name]
        else:
            container.info("Add JSON scenarios under ./scenarios to enable this mode.")
    elif source_mode == "Session L1 artifacts":
        session_dirs = list_engine_session_dirs(sessions_dir)
        l1_files = list((sessions_dir / "l1_event_logs").glob("*.json")) if sessions_dir.exists() else []
        options = ["-- select --"] + [p.name for p in session_dirs] + [f"[log] {p.name}" for p in l1_files]
        choice = container.selectbox("Select L1 source", options, key=f"eval_l1_source_{slot_key}")
        if choice.startswith("[log]"):
            idx = options.index(choice) - 1 - len(session_dirs)
            if 0 <= idx < len(l1_files):
                l1_source_path = l1_files[idx]
        elif choice != "-- select --":
            matching = [p for p in session_dirs if p.name == choice]
            if matching:
                l1_source_path = matching[0]
    elif source_mode == "Session L2 artifacts":
        session_dirs = list_engine_session_dirs(sessions_dir)
        options = ["-- select --"] + [p.name for p in session_dirs]
        choice = container.selectbox("Select session", options, key=f"eval_l2_source_{slot_key}")
        if choice != "-- select --":
            l2_source_path = next((p for p in session_dirs if p.name == choice), None)
    else:  # Session L2 sanitized artifacts
        session_dirs = list_engine_session_dirs(sessions_dir)
        options = ["-- select --"] + [p.name for p in session_dirs]
        choice = container.selectbox("Select session (sanitized)", options, key=f"eval_l2_sanitized_source_{slot_key}")
        if choice != "-- select --":
            l2_source_path = next((p for p in session_dirs if p.name == choice), None)

    def model_select(label: str, key_suffix: str) -> str:
        model_names = list(available_models.keys())
        index = 0
        for i, name in enumerate(model_names):
            if available_models[name] == default_model:
                index = i
                break
        selection = container.selectbox(label, model_names, index=index, key=f"{slot_key}_{key_suffix}")
        return available_models[selection]

    models: Dict[str, Optional[str]] = {}
    language: Optional[str] = None
    if source_mode in {"Scenario", "Session L1 artifacts"}:
        layer1_model = model_select("Layer 1 model", "l1_model")
        layer2_model = model_select("Layer 2 model", "l2_model")
        recon_model = model_select("Reconstruction model", "recon_model")
        language = container.radio("Narrative language", ["en", "he"], horizontal=True, key=f"{slot_key}_lang")
        models = {"layer1": layer1_model, "layer2": layer2_model, "reconstruction": recon_model}
    elif source_mode in {"Session L2 artifacts", "Session L2 sanitized artifacts"}:
        recon_model = model_select("Reconstruction model", "recon_model")
        language = container.radio("Narrative language", ["en", "he"], horizontal=True, key=f"{slot_key}_lang")
        models = {"reconstruction": recon_model}

    if source_mode == "Session L2 artifacts":
        button_label = f"Run L2 stage ({slot_key})"
    elif source_mode == "Session L2 sanitized artifacts":
        button_label = f"Run L2 sanitized stage ({slot_key})"
    else:
        button_label = f"Run slot {slot_key}"

    if container.button(button_label, key=f"run_{slot_key}"):
        result = execute_evaluation_run(
            slot_key=slot_key,
            source_mode=source_mode,
            scenario=scenario,
            l1_source=l1_source_path,
            models=models,
            language=language,
            sessions_dir=sessions_dir,
            l2_source=l2_source_path,
        )
        if result:
            st.session_state.eval_runs[slot_key] = result
            container.success("Run complete.")

    result = st.session_state.eval_runs.get(slot_key)
    if result and result["session"]:
        container.markdown(f"**{result['scenario_name']}** · {len(result['session'].layer1_events)} L1 / {len(result['session'].layer2_events)} L2 events")
        language_display = _format_language_display(getattr(result["session"].metadata, "language", None))
        container.caption(language_display)
        container.markdown(
            "".join(format_layer2_event(evt) for evt in result["session"].layer2_events[:25]),
            unsafe_allow_html=True,
        )
        container.markdown(
            f"<div class='report-container'>{result['narrative']}</div>",
            unsafe_allow_html=True,
        )


def render_comparison():
    run_a = st.session_state.eval_runs.get("A")
    run_b = st.session_state.eval_runs.get("B")
    if not run_a or not run_b:
        return
    st.subheader("🔍 Run Comparison")
    metrics = [
        ("Scenario", run_a["scenario_name"], run_b["scenario_name"]),
        ("Layer 1 events", len(run_a["session"].layer1_events), len(run_b["session"].layer1_events)),
        ("Layer 2 events", len(run_a["session"].layer2_events), len(run_b["session"].layer2_events)),
        ("Language", run_a["language"], run_b["language"]),
        (
            "Models (L1/L2/Recon)",
            f"{run_a['models'].get('layer1', '—')} / {run_a['models'].get('layer2', '—')} / {run_a['models'].get('reconstruction', '—')}",
            f"{run_b['models'].get('layer1', '—')} / {run_b['models'].get('layer2', '—')} / {run_b['models'].get('reconstruction', '—')}",
        ),
    ]
    for label, left, right in metrics:
        st.markdown(f"- **{label}:** {left} ↔ {right}")


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main():
    _init_session_state()
    config = get_config()
    available_models = get_available_models()

    base_dir = Path(__file__).parent
    scenarios_dir = base_dir / config.scenarios_directory
    sessions_dir = resolve_sessions_directory()

    for directory in (
        scenarios_dir,
        sessions_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    st.title("👁️ IRIS Scenario Lab")
    #st.subheader("<span class='form-label'>Privacy-first classroom observation · Demo + Evaluation</span>", unsafe_allow_html=True)    
    st.subheader("Privacy-first classroom observation · Demo + Evaluation") 

    tabs = st.tabs(["Interactive Demo", "Evaluation Dashboard"])
    with tabs[0]:
        render_demo_tab(
            scenarios_dir=scenarios_dir,
            sessions_dir=sessions_dir,
            available_models=available_models,
            config_model=config.default_model,
        )
    with tabs[1]:
        render_session_browser(sessions_dir)
        col_a, col_b = st.columns(2)
        with col_a:
            render_eval_slot("A", scenarios_dir, sessions_dir, available_models, config.default_model)
        with col_b:
            render_eval_slot("B", scenarios_dir, sessions_dir, available_models, config.default_model)
        render_comparison()


if __name__ == "__main__":
    main()
