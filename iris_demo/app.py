"""
IRIS Demo App v3.1

Streamlit interface for demonstrating the IRIS system.

Features:
- Load scenarios from JSON or write custom descriptions
- Real-time streaming display of Layer 1 and Layer 2 events
- Three report types:
  1. SLP/Clinical Report
  2. Anonymous Story Reconstruction
  3. Social Story Generation

Configuration:
- API keys: Read from .env file
- Models: Configured in config.yaml
"""

import streamlit as st
import json
import yaml
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add parent to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    # Config and LLM
    get_config, get_available_models, AISUITE_AVAILABLE,
    
    # Generator
    Scenario, EventGenerator, load_scenarios_from_directory,
    
    # Schema
    Session, SessionMetadata,
    SpeechEvent, AmbientAudioEvent, ProximityEvent, GazeEvent,
    PostureEvent, ObjectEvent,
    BehavioralEvent, InteractionEvent, ContextEvent,
    
    # Storage
    save_session, load_session, list_saved_sessions, serialize_session,
    
    # Reports
    generate_report, REPORT_TYPES,
    
    # Reconstruction
    L1EventLog, ReconstructionPipeline, load_l1_event_log,
)


# =============================================================================
# SCENARIO PERSISTENCE
# =============================================================================

def save_scenario(scenario: Scenario, directory: Path) -> str:
    """Save a scenario to JSON file. Returns filename."""
    import json
    
    directory.mkdir(exist_ok=True)
    
    # Create safe filename
    safe_name = "".join(c if c.isalnum() else "_" for c in scenario.name)
    filename = f"{safe_name}.json"
    filepath = directory / filename
    
    # Don't overwrite existing scenarios with same name
    counter = 1
    while filepath.exists():
        filename = f"{safe_name}_{counter}.json"
        filepath = directory / filename
        counter += 1
    
    # Serialize scenario
    data = {
        "name": scenario.name,
        "description": scenario.description,
        "duration_minutes": scenario.duration_minutes,
        "num_children": scenario.num_children,
        "num_adults": scenario.num_adults,
    }
    if scenario.focus_children:
        data["focus_children"] = scenario.focus_children
    if scenario.key_moments:
        data["key_moments"] = scenario.key_moments
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return filename


# =============================================================================
# REPORT PERSISTENCE
# =============================================================================

def save_report(
    model: str,
    report_text: str, 
    report_type: str, 
    session_id: str, 
    scenario_name: str,
    directory: Path
) -> str:
    """Save a report to file. Returns filename."""
    directory.mkdir(exist_ok=True)
    
    # Create filename with session info
    safe_scenario = "".join(c if c.isalnum() else "_" for c in scenario_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{safe_scenario}_{report_type}_{timestamp}.md"
    filepath = directory / filename
    
    # Add metadata header to report
    header = f"""---
report_type: {report_type}
session_id: {session_id}
scenario: {scenario_name}
generated: {timestamp}
model: {model}
---

"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header + report_text)
    
    return filename


def list_saved_reports(directory: Path) -> list:
    """List all saved reports with metadata."""
    reports = []
    
    if not directory.exists():
        return reports
    
    for filepath in sorted(directory.glob("*.md"), reverse=True):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse YAML frontmatter if present
            metadata = {
                "path": str(filepath),
                "filename": filepath.name,
            }
            
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    if frontmatter:
                        metadata.update(frontmatter)
                    metadata["content"] = parts[2].strip()
                else:
                    metadata["content"] = content
            else:
                metadata["content"] = content
            
            reports.append(metadata)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    return reports


def normalize_metadata_value(value) -> str:
    """Convert metadata values (datetime/str/None) to safe string."""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value is not None else ""


def build_scenario_archive(saved_sessions: list, saved_reports: list) -> dict:
    """Group sessions and reports by scenario for easier browsing."""
    archive = {}
    for session in saved_sessions:
        scenario_name = session.get("scenario_name") or "Unknown Scenario"
        entry = archive.setdefault(
            scenario_name,
            {"sessions": [], "reports_by_session": defaultdict(list)}
        )
        entry["sessions"].append(session)
    
    for report in saved_reports:
        scenario_name = report.get("scenario", "Unknown Scenario")
        entry = archive.setdefault(
            scenario_name,
            {"sessions": [], "reports_by_session": defaultdict(list)}
        )
        session_key = report.get("session_id") or "__unlinked__"
        entry["reports_by_session"][session_key].append(report)
    
    # Convert defaultdicts to normal dicts for downstream usage
    for entry in archive.values():
        entry["reports_by_session"] = dict(entry["reports_by_session"])
    
    return archive


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="IRIS Demo v3.1",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Event cards */
    .event-card {
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 8px;
        font-size: 0.85em;
        border-left: 5px solid;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Layer 1 - Audio events */
    .event-speech { background: #e3f2fd; border-color: #1976d2; }
    .event-ambient { background: #e1f5fe; border-color: #0288d1; }
    
    /* Layer 1 - Video events */
    .event-proximity { background: #f3e5f5; border-color: #7b1fa2; }
    .event-gaze { background: #e0f7fa; border-color: #00838f; }
    .event-posture { background: #fff3e0; border-color: #ef6c00; }
    .event-object { background: #fff8e1; border-color: #f9a825; }
    
    /* Layer 2 - Inferred events */
    .event-behavioral { background: #fce4ec; border-color: #c2185b; }
    .event-interaction { background: #e8f5e9; border-color: #388e3c; }
    .event-context { background: #ede7f6; border-color: #512da8; }
    
    /* Phase headers */
    .phase-header {
        padding: 8px 16px;
        margin: 16px 0 8px 0;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.9em;
    }
    .phase-layer1 { background: #e3f2fd; color: #1565c0; }
    .phase-layer2 { background: #fce4ec; color: #c2185b; }
    
    /* Climate indicators */
    .climate-calm { color: #4caf50; }
    .climate-focused { color: #2196f3; }
    .climate-energetic { color: #ff9800; }
    .climate-restless { color: #ff5722; }
    .climate-chaotic { color: #f44336; font-weight: bold; }
    .climate-tense { color: #9c27b0; }
    
    /* Report sections */
    .report-container {
        background: #fafafa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #e0e0e0;
    }
    
    /* Timestamp */
    .timestamp {
        color: #757575;
        font-size: 0.8em;
        font-family: monospace;
    }
    
    /* Actor labels */
    .actor-child { color: #1976d2; font-weight: 600; }
    .actor-adult { color: #388e3c; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

if "session" not in st.session_state:
    st.session_state.session = None
if "layer1_events" not in st.session_state:
    st.session_state.layer1_events = []
if "layer2_events" not in st.session_state:
    st.session_state.layer2_events = []
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "loaded_report" not in st.session_state:
    st.session_state.loaded_report = None


# =============================================================================
# EVENT FORMATTING
# =============================================================================

def format_timestamp(event) -> str:
    """Format event timestamp as offset from session start."""
    # Get base timestamp from session metadata if available
    base = None
    
    # Try to get from loaded session metadata first
    if "session" in st.session_state and st.session_state.session:
        base = st.session_state.session.metadata.start_time
    # Fall back to first event in layer1_events
    elif st.session_state.layer1_events:
        base = st.session_state.layer1_events[0].timestamp
    
    if base and event.timestamp:
        offset = (event.timestamp - base).total_seconds()
        mins = int(offset // 60)
        secs = int(offset % 60)
        return f"{mins:02d}:{secs:02d}"
    return "00:00"


def format_actor(actor) -> str:
    """Format actor with appropriate styling."""
    css_class = "actor-child" if actor.role.value == "child" else "actor-adult"
    return f'<span class="{css_class}">{actor.id}</span>'


def format_layer1_event(event) -> str:
    """Format a Layer 1 event as HTML."""
    ts = format_timestamp(event)
    
    if isinstance(event, SpeechEvent):
        target = f" → {event.target.value}" if event.target.value != "unknown" else ""
        transcription = event.transcription or f"[{event.vocal_type.value}]"
        prosody_info = ""
        if event.prosody and event.prosody.pitch_contour:
            prosody_info = f" 📊 {event.prosody.pitch_contour}"
        echo = " 🔄" if event.is_echolalia_candidate else ""
        
        return f"""
        <div class="event-card event-speech">
            <span class="timestamp">[{ts}]</span> 🎤 
            {format_actor(event.speaker)}{target}: "{transcription}"
            <br><small>complexity: {event.complexity.value} | {event.duration_ms}ms{prosody_info}{echo}</small>
        </div>
        """
    
    elif isinstance(event, AmbientAudioEvent):
        return f"""
        <div class="event-card event-ambient">
            <span class="timestamp">[{ts}]</span> 🔊 
            <b>{event.sound_type}</b> ({event.intensity.value} intensity, {event.duration_ms}ms)
        </div>
        """
    
    elif isinstance(event, ProximityEvent):
        speed = f", {event.movement_speed}" if event.movement_speed else ""
        return f"""
        <div class="event-card event-proximity">
            <span class="timestamp">[{ts}]</span> 📍 
            {format_actor(event.actor)} {event.change_type.value} {format_actor(event.target)}
            <br><small>proximity: {event.proximity_level.value}{speed}</small>
        </div>
        """
    
    elif isinstance(event, GazeEvent):
        target = ""
        if event.target_actor:
            target = f" at {format_actor(event.target_actor)}"
        elif event.target_object:
            target = f" at {event.target_object}"
        mutual = " 👀 mutual" if event.is_mutual else ""
        sustained = " (sustained)" if event.is_sustained else ""
        fleeting = " (fleeting)" if event.is_fleeting else ""
        
        return f"""
        <div class="event-card event-gaze">
            <span class="timestamp">[{ts}]</span> 👁️ 
            {format_actor(event.actor)} gaze {event.direction.value}{target}
            <br><small>{event.duration_ms}ms{mutual}{sustained}{fleeting}</small>
        </div>
        """
    
    elif isinstance(event, PostureEvent):
        repetitive = " 🔁 repetitive" if event.is_repetitive else ""
        return f"""
        <div class="event-card event-posture">
            <span class="timestamp">[{ts}]</span> 🧍 
            {format_actor(event.actor)}: {event.posture.value}, {event.movement.value}
            <br><small>orientation: {event.orientation.value} | intensity: {event.movement_intensity.value}{repetitive}</small>
        </div>
        """
    
    elif isinstance(event, ObjectEvent):
        shared = f" with {format_actor(event.shared_with)}" if event.shared_with else ""
        return f"""
        <div class="event-card event-object">
            <span class="timestamp">[{ts}]</span> 🎯 
            {format_actor(event.actor)} {event.action.value} <b>{event.object_type}</b>{shared}
        </div>
        """
    
    return f"<div class='event-card'>Unknown event type</div>"


def format_layer2_event(event) -> str:
    """Format a Layer 2 event as HTML."""
    ts = format_timestamp(event)
    
    if isinstance(event, BehavioralEvent):
        emotion_emoji = {
            "calm": "😌", "happy": "😊", "excited": "🤩", "anxious": "😰",
            "frustrated": "😤", "sad": "😢", "angry": "😠", "overwhelmed": "😫",
            "withdrawn": "😶", "neutral": "😐", "unclear": "❓"
        }.get(event.apparent_emotion.value, "❓")
        
        trigger_info = ""
        if event.trigger_description:
            trigger_info = f" ← {event.trigger_description}"
        elif event.trigger.value != "unknown":
            trigger_info = f" ← {event.trigger.value}"
        
        regulation = ""
        if event.regulation_effective is not None:
            regulation = " ✓ effective" if event.regulation_effective else " ✗ not effective"
        
        return f"""
        <div class="event-card event-behavioral">
            <span class="timestamp">[{ts}]</span> 🧠 
            {format_actor(event.actor)}: <b>{event.category.value}</b>
            <br>{event.description}
            <br><small>{emotion_emoji} {event.apparent_emotion.value} | 
            intensity: {event.intensity.value}{trigger_info}{regulation}
            <br>confidence: {event.confidence:.0%} | sources: {len(event.source_event_ids)}</small>
        </div>
        """
    
    elif isinstance(event, InteractionEvent):
        quality_emoji = {
            "successful": "✅", "partial": "🔶", "unsuccessful": "❌", "ongoing": "🔄"
        }.get(event.quality.value, "❓")
        
        facilitated = " 👨‍🏫 adult facilitated" if event.adult_facilitated else ""
        
        return f"""
        <div class="event-card event-interaction">
            <span class="timestamp">[{ts}]</span> 🤝 
            {format_actor(event.initiator)} → {format_actor(event.recipient)}: <b>{event.interaction_type.value}</b>
            <br>{event.description}
            <br><small>{quality_emoji} {event.quality.value} | 
            reciprocity: {event.reciprocity_level or 'n/a'}{facilitated}
            <br>confidence: {event.confidence:.0%}</small>
        </div>
        """
    
    elif isinstance(event, ContextEvent):
        climate_class = f"climate-{event.classroom_climate.value}"
        transition = " 🔀 TRANSITION" if event.is_transition else ""
        
        return f"""
        <div class="event-card event-context">
            <span class="timestamp">[{ts}]</span> 🏫 
            <b>{event.activity_type.value}</b> @ {event.primary_zone.value}
            <br><small>climate: <span class="{climate_class}">{event.classroom_climate.value.upper()}</span> | 
            noise: {event.noise_level.value} | adults: {event.adults_present}{transition}</small>
        </div>
        """
    
    return f"<div class='event-card'>Unknown event type</div>"


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.title("👁️ IRIS Demo v7.83")
    st.caption("Intelligent Room Insight System - Privacy-First Classroom Observation")
    
    # Load config
    config = get_config()
    available_models = get_available_models()
    base_dir = Path(__file__).parent
    scenarios_dir = base_dir / config.scenarios_directory
    sessions_dir = base_dir / config.sessions_directory
    reports_dir = base_dir / config.reports_directory
    
    for directory in (scenarios_dir, sessions_dir, reports_dir):
        directory.mkdir(exist_ok=True)
    
    saved_sessions = list_saved_sessions(str(sessions_dir))
    saved_reports = list_saved_reports(reports_dir)
    scenario_archive = build_scenario_archive(saved_sessions, saved_reports)
    
    # Check for API key
    api_key_present = os.environ.get("ANTHROPIC_API_KEY") is not None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API key status
        #if api_key_present:
        #    st.success("✓ API key loaded from .env")
        #else:
        #    st.error("⚠️ No ANTHROPIC_API_KEY found in .env")
        #    st.info("Create a .env file with:\n```\nANTHROPIC_API_KEY=sk-ant-...\n```")
        
        # Model selection
        model_names = list(available_models.keys())
        if not model_names:
            st.warning("No models available")
            selected_model = None
        else:
            # Find default
            default_idx = 0
            for i, name in enumerate(model_names):
                if available_models[name] == config.default_model:
                    default_idx = i
                    break
            
            selected_model_name = st.selectbox(
                "Model",
                model_names,
                index=default_idx,
            )
            
            selected_model = available_models[selected_model_name]
            
            # Show if it's aisuite
            if ":" in selected_model:
                st.info("ℹ️ Uses aisuite (batch mode)")
            else:
                st.info("ℹ️ Uses Anthropic SDK (streaming)")
        
        if not AISUITE_AVAILABLE and any(":" in m for m in available_models.values()):
            st.warning("aisuite not installed (requires Python 3.10+)")
        else:
            st.success("✓ aisuite available" if AISUITE_AVAILABLE else "✓ Using Anthropic SDK")
        
        st.divider()
        
        # Main tabs in sidebar
        sidebar_tabs = st.tabs(["Generate", "Reconstruct"])
        
        with sidebar_tabs[0]:  # Generate tab
            # Scenario selection
            st.header("📋 Scenario")
            
            scenario_mode = st.radio("Source", ["Load from file", "Custom text"])
            
            scenario = None
            is_custom_scenario = False
            
            if scenario_mode == "Load from file":
                scenarios = load_scenarios_from_directory(str(scenarios_dir))
                if scenarios:
                    selected_name = st.selectbox("Select scenario", list(scenarios.keys()))
                    scenario = scenarios[selected_name]
                    
                    with st.expander("Scenario details"):
                        st.write(f"**Duration:** {scenario.duration_minutes} min")
                        st.write(f"**Children:** {scenario.num_children}")
                        st.write(f"**Adults:** {scenario.num_adults}")
                        st.write(f"**Description:**")
                        st.write(scenario.description)
                else:
                    st.info("No scenarios found in scenarios/ folder")
            
            else:  # Custom text
                custom_name = st.text_input("Scenario name", "Custom Scenario")
                custom_duration = st.slider("Duration (minutes)", 5, 20, 10)
                custom_children = st.slider("Number of children", 2, 10, 5)
                custom_adults = st.slider("Number of adults", 1, 4, 2)
                custom_description = st.text_area(
                    "Describe the scenario",
                    placeholder="Describe what happens in the classroom session...",
                    height=150
                )
                
                if custom_description:
                    scenario = Scenario(
                        name=custom_name,
                        description=custom_description,
                        duration_minutes=custom_duration,
                        num_children=custom_children,
                        num_adults=custom_adults,
                    )
                    is_custom_scenario = True
                    
                    # Save custom scenario button
                    if st.button("💾 Save Scenario", use_container_width=True):
                        try:
                            filename = save_scenario(scenario, scenarios_dir)
                            st.success(f"Saved: {filename}")
                        except Exception as e:
                            st.error(f"Error saving scenario: {e}")
            
            st.divider()
            
            st.header("📚 Scenario Archive")
            if scenario_archive:
                with st.expander("Browse saved sessions & reports", expanded=False):
                    for scenario_name in sorted(scenario_archive.keys()):
                        entry = scenario_archive[scenario_name]
                        scenario_sessions = sorted(
                            entry["sessions"],
                            key=lambda s: s.get("start_time") or "",
                            reverse=True
                        )
                        report_groups = entry["reports_by_session"]
                        total_reports = sum(len(r) for r in report_groups.values())
                        
                        st.markdown(f"**{scenario_name}**")
                        st.caption(f"{len(scenario_sessions)} session(s) · {total_reports} report(s)")
                        
                        if not scenario_sessions and total_reports == 0:
                            st.write("No saved data yet.")
                        
                        for session_info in scenario_sessions:
                            session_id = session_info.get("session_id") or "unknown"
                            start_display = normalize_metadata_value(session_info.get("start_time"))[:16] or "unknown"
                            layer_counts = f"{session_info.get('layer1_count', 0)} L1 / {session_info.get('layer2_count', 0)} L2"
                            people_info = ""
                            if session_info.get("num_children") is not None:
                                people_info = f" · {session_info.get('num_children')} children"
                            session_label = f"- Session {session_id}\n  {start_display} · {layer_counts}{people_info}"
                            st.markdown(session_label)
                            
                            load_key = f"sidebar_archive_load_{session_info['path']}"
                            if st.button("Load", key=load_key):
                                try:
                                    loaded = load_session(session_info['path'])
                                    st.session_state.loaded_report = None
                                    st.session_state.session = loaded
                                    st.session_state.layer1_events = loaded.layer1_events
                                    st.session_state.layer2_events = loaded.layer2_events
                                    st.success(f"Loaded: {loaded.metadata.scenario_name}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error loading session: {e}")
                            
                            session_reports = report_groups.get(session_info.get("session_id")) or []
                            for report_meta in sorted(
                                session_reports,
                                key=lambda r: normalize_metadata_value(r.get("generated")),
                                reverse=True
                            ):
                                report_desc = (
                                    f"{report_meta.get('report_type', '?')} · "
                                    f"{normalize_metadata_value(report_meta.get('generated'))[:16]}"
                                )
                                open_key = f"sidebar_open_report_{report_meta['filename']}"
                                if st.button(f"Open report: {report_desc}", key=open_key):
                                    st.session_state.loaded_report = report_meta
                                    st.rerun()
                        
                        orphan_reports = report_groups.get("__unlinked__", [])
                        if orphan_reports:
                            st.markdown("_Reports without linked session:_")
                            for report_meta in sorted(
                                orphan_reports,
                                key=lambda r: normalize_metadata_value(r.get("generated")),
                                reverse=True
                            ):
                                report_desc = (
                                    f"{report_meta.get('report_type', '?')} · "
                                    f"{normalize_metadata_value(report_meta.get('generated'))[:16]}"
                                )
                                orphan_key = f"sidebar_open_orphan_report_{report_meta['filename']}"
                                if st.button(f"Open report: {report_desc}", key=orphan_key):
                                    st.session_state.loaded_report = report_meta
                                    st.rerun()
            else:
                st.info("No saved sessions or reports yet.")
            
            st.divider()
            
            # Generate button
            generate_disabled = not api_key_present or scenario is None or selected_model is None
            if st.button("🚀 Generate Events", disabled=generate_disabled, use_container_width=True):
                st.session_state.is_generating = True
                st.session_state.layer1_events = []
                st.session_state.layer2_events = []
                st.session_state.session = None
        
        with sidebar_tabs[1]:  # Reconstruct tab
            st.header("🔬 Scenario Reconstruction")
            st.markdown("""
            Reconstruct scenarios from saved Layer 1 event logs.
            
            **Pipeline:**
            1. Load L1 event log
            2. Send L1 events to LLM → get L2 inferences
            3. Send L2 events to LLM → reconstruct scenario
            """)
            
            # Find saved L1 logs (now stored under a dedicated subfolder)
            l1_logs_dir = sessions_dir / "l1_event_logs"
            l1_logs = []
            if l1_logs_dir.exists():
                l1_logs = sorted(l1_logs_dir.glob("*_L1_events_*.json"))
            
            if l1_logs:
                selected_log = st.selectbox(
                    "Select L1 event log",
                    l1_logs,
                    format_func=lambda p: p.name
                )
                
                st.caption(f"File: {selected_log.name}")
                
                if st.button("📊 Run Reconstruction Pipeline", use_container_width=True):
                    with st.spinner("Loading L1 event log..."):
                        try:
                            l1_log = load_l1_event_log(str(selected_log))
                            
                            st.info(f"Loaded: {l1_log.scenario_name}")
                            st.caption(f"{len(l1_log.l1_events)} L1 events, {l1_log.duration_seconds}s duration")
                            
                            # Run reconstruction pipeline
                            pipeline = ReconstructionPipeline(model_id=selected_model)
                            
                            result = pipeline.full_reconstruction_pipeline(l1_log)
                            
                            # Display results
                            st.success("✓ Reconstruction complete!")
                            
                            # Privacy Validation Section
                            st.markdown("### 🔐 Privacy Validation")
                            if 'validation_report' in result and result['validation_report']:
                                report = result['validation_report']
                                
                                # Create three columns for metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Events Validated", report.events_validated)
                                with col2:
                                    st.metric("Events Passed", report.events_passed)
                                with col3:
                                    st.metric("Success Rate", f"{report.validation_success_rate * 100:.1f}%")
                                
                                # Show status
                                if report.events_failed == 0:
                                    st.success(f"✅ All L2 events pass privacy validation - No PII detected")
                                else:
                                    st.warning(f"⚠️ {report.events_failed} event(s) failed privacy validation")
                                    with st.expander("View Failed Events", expanded=True):
                                        for event_id, details in zip(report.failed_event_ids, report.failed_event_details):
                                            st.write(f"**{event_id}**: {details}")
                            
                            st.markdown("### Layer 2 Reconstruction")
                            st.metric("L2 Events Generated", len(result['l2_events']))
                            
                            with st.expander("View L2 Events", expanded=False):
                                for evt in result['l2_events']:
                                    if hasattr(evt, 'category'):  # BehavioralEvent
                                        st.write(f"**{evt.category.value}**: {evt.description}")
                                    elif hasattr(evt, 'initiator'):  # InteractionEvent
                                        st.write(f"**Interaction**: {evt.initiator.id} → {evt.recipient.id}")
                                    elif hasattr(evt, 'activity_type'):  # ContextEvent
                                        st.write(f"**Context**: {evt.activity_type.value}")
                            
                            st.markdown("### Scenario Reconstruction")
                            st.markdown(result['reconstructed_scenario'])
                            
                            # Download results
                            results_json = json.dumps({
                                "original_scenario": {
                                    "name": l1_log.scenario_name,
                                    "description": l1_log.scenario_description,
                                },
                                "l2_events_count": len(result['l2_events']),
                                "reconstructed_scenario": result['reconstructed_scenario'],
                            }, ensure_ascii=False, indent=2)
                            
                            st.download_button(
                                "⬇️ Download Reconstruction Results",
                                results_json,
                                file_name=f"reconstruction_{l1_log.log_id}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        except Exception as e:
                            st.error(f"Reconstruction error: {e}")
                            import traceback
                            st.code(traceback.format_exc())
            else:
                st.info("No L1 event logs found. Generate events first to create logs.")
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("📊 Event Stream")
        
        # Event display container
        event_container = st.container()
        
        # Generation logic
        if st.session_state.is_generating and scenario and selected_model:
            with event_container:
                try:
                    generator = EventGenerator(model=selected_model)
                    
                    # Phase 1: Layer 1 events
                    st.markdown('<div class="phase-header phase-layer1">🎬 LAYER 1: Raw Sensor Events</div>', 
                               unsafe_allow_html=True)
                    
                    layer1_placeholder = st.empty()
                    layer1_events = []
                    
                    for event in generator.generate_layer1_events(scenario):
                        layer1_events.append(event)
                        
                        # Update display
                        html = "".join([format_layer1_event(e) for e in layer1_events[-20:]])
                        layer1_placeholder.markdown(html, unsafe_allow_html=True)
                    
                    st.session_state.layer1_events = layer1_events
                    st.success(f"✓ Generated {len(layer1_events)} Layer 1 events")
                    
                    # Phase 2: Layer 2 events
                    st.markdown('<div class="phase-header phase-layer2">🧠 LAYER 2: LLM Inference</div>', 
                               unsafe_allow_html=True)
                    
                    layer2_placeholder = st.empty()
                    layer2_events = []
                    
                    for event in generator.generate_layer2_events(layer1_events, scenario):
                        layer2_events.append(event)
                        
                        html = "".join([format_layer2_event(e) for e in layer2_events])
                        layer2_placeholder.markdown(html, unsafe_allow_html=True)
                    
                    st.session_state.layer2_events = layer2_events
                    st.success(f"✓ Generated {len(layer2_events)} Layer 2 events")
                    
                    # Build session object
                    import uuid
                    
                    session_obj = Session(
                        metadata=SessionMetadata(
                            session_id=uuid.uuid4().hex[:12],
                            start_time=layer1_events[0].timestamp if layer1_events else datetime.now(),
                            end_time=layer1_events[-1].timestamp if layer1_events else datetime.now(),
                            scenario_name=scenario.name,
                            scenario_description=scenario.description,
                            num_children=scenario.num_children,
                            num_adults=scenario.num_adults,
                            layer2_llm_model=selected_model,
                        ),
                        actors=list(generator.actors.values()),
                    )
                    
                    # Populate event lists
                    for event in layer1_events:
                        if isinstance(event, SpeechEvent):
                            session_obj.speech_events.append(event)
                        elif isinstance(event, AmbientAudioEvent):
                            session_obj.ambient_audio_events.append(event)
                        elif isinstance(event, ProximityEvent):
                            session_obj.proximity_events.append(event)
                        elif isinstance(event, GazeEvent):
                            session_obj.gaze_events.append(event)
                        elif isinstance(event, PostureEvent):
                            session_obj.posture_events.append(event)
                        elif isinstance(event, ObjectEvent):
                            session_obj.object_events.append(event)
                    
                    for event in layer2_events:
                        if isinstance(event, BehavioralEvent):
                            session_obj.behavioral_events.append(event)
                        elif isinstance(event, InteractionEvent):
                            session_obj.interaction_events.append(event)
                        elif isinstance(event, ContextEvent):
                            session_obj.context_events.append(event)
                    
                    st.session_state.session = session_obj
                    st.session_state.loaded_report = None
                    
                    # Auto-save custom scenario if not already saved
                    if is_custom_scenario:
                        try:
                            filename = save_scenario(scenario, scenarios_dir)
                            st.info(f"📋 Custom scenario saved: {filename}")
                        except Exception as e:
                            st.warning(f"Could not save scenario: {e}")
                    
                    # Auto-save session
                    if config.sessions_auto_save:
                        safe_name = "".join(c if c.isalnum() else "_" for c in scenario.name)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{safe_name}_{timestamp}.json"
                        filepath = sessions_dir / filename
                        
                        save_session(session_obj, str(filepath))
                        st.success(f"💾 Session saved: {filename}")
                    
                except Exception as e:
                    st.error(f"Generation error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                
                finally:
                    st.session_state.is_generating = False
        
        # Display existing events if not generating
        elif st.session_state.layer1_events or st.session_state.layer2_events:
            with event_container:
                if st.session_state.layer1_events:
                    st.markdown('<div class="phase-header phase-layer1">🎬 LAYER 1: Raw Sensor Events</div>', 
                               unsafe_allow_html=True)
                    
                    # Show event type filter
                    event_types = ["All", "Speech", "Ambient", "Proximity", "Gaze", "Posture", "Object"]
                    selected_type = st.selectbox("Filter Layer 1", event_types, key="l1_filter")
                    
                    filtered = st.session_state.layer1_events
                    if selected_type != "All":
                        type_map = {
                            "Speech": SpeechEvent, "Ambient": AmbientAudioEvent,
                            "Proximity": ProximityEvent, "Gaze": GazeEvent,
                            "Posture": PostureEvent, "Object": ObjectEvent
                        }
                        filtered = [e for e in filtered if isinstance(e, type_map[selected_type])]
                    
                    html = "".join([format_layer1_event(e) for e in filtered])
                    st.markdown(html, unsafe_allow_html=True)
                
                if st.session_state.layer2_events:
                    st.markdown('<div class="phase-header phase-layer2">🧠 LAYER 2: LLM Inference</div>', 
                               unsafe_allow_html=True)
                    
                    html = "".join([format_layer2_event(e) for e in st.session_state.layer2_events])
                    st.markdown(html, unsafe_allow_html=True)
        
        else:
            st.info("Select a scenario and click 'Generate Events' to start")
    
    with col2:
        st.header("📝 Reports")
        
        # Display loaded report
        if "loaded_report" in st.session_state and st.session_state.loaded_report:
            report_meta = st.session_state.loaded_report
            st.markdown("---")
            st.markdown(f"### 📄 {report_meta.get('report_type', 'Report')}")
            generated_value = normalize_metadata_value(report_meta.get('generated', 'Unknown'))
            st.caption(f"Scenario: {report_meta.get('scenario', 'Unknown')} | Generated: {generated_value[:19]}")
            st.markdown(f'<div class="report-container">{report_meta.get("content", "")}</div>', 
                       unsafe_allow_html=True)
            
            col_clear, col_download = st.columns(2)
            with col_clear:
                if st.button("✖️ Close", use_container_width=True):
                    st.session_state.loaded_report = None
                    st.rerun()
            with col_download:
                st.download_button(
                    "⬇️ Download",
                    report_meta.get("content", ""),
                    file_name=report_meta.get("filename", "report.md"),
                    mime="text/markdown",
                    use_container_width=True
                )
        
        st.divider()
        
        if st.session_state.session:
            session = st.session_state.session
            
            # Stats summary
            with st.expander("📊 Session Statistics", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Layer 1 Events", len(session.layer1_events))
                    st.metric("Speech Events", len(session.speech_events))
                    st.metric("Gaze Events", len(session.gaze_events))
                with col_b:
                    st.metric("Layer 2 Events", len(session.layer2_events))
                    st.metric("Behavioral", len(session.behavioral_events))
                    st.metric("Interactions", len(session.interaction_events))
                
                # Export session JSON
                session_json = json.dumps(serialize_session(session), ensure_ascii=False, indent=2)
                st.download_button(
                    "⬇️ Export Session JSON",
                    session_json,
                    file_name=f"iris_session_{session.metadata.session_id}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            st.divider()
            
            # Report generation
            report_type_names = {
                "slp_clinical": "SLP/Clinical Report",
                "anonymous_story": "Anonymous Story",
                "social_story": "Social Story",
            }
            
            report_type = st.radio(
                "Report Type",
                list(report_type_names.keys()),
                format_func=lambda x: report_type_names[x],
                help="Select the type of report to generate"
            )
            
            if st.button("📄 Generate Report", use_container_width=True):
                with st.spinner("Generating report..."):
                    try:
                        report = generate_report(
                            session, 
                            report_type,
                            model=selected_model
                        )
                        
                        # Auto-save report
                        if config.reports_auto_save:
                            st.info("Auto-saving report...")
                            try:
                                filename = save_report(
                                    selected_model,
                                    report,
                                    report_type,
                                    session.metadata.session_id,
                                    session.metadata.scenario_name,
                                    reports_dir
                                )
                                st.success(f"💾 Report saved: {filename}")
                            except Exception as e:
                                st.warning(f"Could not save report: {e}")
                        
                        st.markdown("---")
                        st.markdown(f"### {report_type_names[report_type]}")
                        st.markdown(f'<div class="report-container">{report}</div>', 
                                   unsafe_allow_html=True)
                        
                        # Download button
                        st.download_button(
                            "⬇️ Download Report",
                            report,
                            file_name=f"iris_report_{report_type}.md",
                            mime="text/markdown"
                        )
                        
                    except Exception as e:
                        st.error(f"Report generation error: {e}")
        
        else:
            st.info("Generate events first to enable reports")


if __name__ == "__main__":
    main()
