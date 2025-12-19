"""
Test: Scenario Reconstruction Pipeline

Tests the full reconstruction pipeline:
1. Load scenario
2. Generate and save L1 event log via ObservationEngine
3. Produce/sanitize L2 events and reconstruct with the engine

This validates that we can reconstruct a scenario from raw events,
measuring the fidelity of our event system.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src import (
    Config,
    Scenario,
    ObservationEngine,
    L1EventLog,
    load_l1_log_from_session,
    load_l2_event_log,
    get_available_models,
)
from src.core.paths import resolve_sessions_directory

CONFIG = Config.load()
SESSIONS_DIR = resolve_sessions_directory(CONFIG.sessions_directory)
from src.core.layer2 import Layer2Sanitizer
from src.core.session import Layer2Batch
from src.core.storage import deserialize_event, serialize_event
from dataclasses import asdict
import json

def test_l1_event_generation(model: str = None, lan: str = "en", scenario: Scenario = None):
    """
    Test generating Layer 1 events from a scenario and saving the log.
    """
    print("\n" + "="*80)
    print("TEST: LAYER 1 EVENT GENERATION")
    print("="*80)
    
    # Create a test scenario
    scenario = scenario or Scenario(
        name="Test_Scenario_L1_Generation",
        description="A group of children are playing with toys. One child shares a toy with another.",
        duration_minutes=5,
        num_children=4,
        num_adults=1,
    )
    print(f"Scenario: {scenario.name}")
    print(f"Description: {scenario.description[:60]}...")
    
    # Generate Layer 1 events via the core engine
    engine = ObservationEngine(model_id = model, artifact_lan=lan)
    handle = engine.create_session(scenario)
    layer1_batch = engine.run_layer1(handle)
    l1_events = layer1_batch.events
    print(f"Generated {len(l1_events)} Layer 1 events")
    
    # Create and save L1 log
    log_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    duration_seconds = int((l1_events[-1].timestamp - l1_events[0].timestamp).total_seconds()) if l1_events else 0
    
    l1_log = L1EventLog(
        log_id=log_id,
        scenario_name=scenario.name,
        scenario_description=scenario.description,
        timestamp_created=datetime.now(),
        duration_seconds=duration_seconds,
        num_children=scenario.num_children,
        num_adults=scenario.num_adults,
        l1_events=l1_events,
    )
    
    session_dir = SESSIONS_DIR / handle.session_id
    print(f"Session artifacts stored in: {session_dir}")
    
    return l1_log, handle.session_id


def save_scenario_to_file(scenario: Scenario, out_path: str):
    """Save a Scenario dataclass to a JSON file for reuse."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(scenario), f, ensure_ascii=False, indent=2)
    return str(out_path)


def _resolve_path(path_value: str) -> Path:
    raw_path = Path(path_value).expanduser()
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        (Path.cwd() / raw_path),
        (project_root / raw_path),
        (project_root.parent / raw_path),
    ]

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    # Fallback to project_root-relative even if it doesn't exist yet
    return (project_root / raw_path).resolve()


def _default_scenario_path(name: str) -> Path:
    safe_name = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in name) or "scenario"
    scenarios_dir = SESSIONS_DIR / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    return scenarios_dir / f"{safe_name}.json"


def _scenario_from_argument(raw: str) -> Scenario:
    """Parse --scenario input (JSON or NAME:::DESCRIPTION)."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return Scenario.from_dict(data)
    except json.JSONDecodeError:
        pass
    if ":::" in raw:
        name, description = raw.split(":::", 1)
        return Scenario(name=name.strip(), description=description.strip())
    raise ValueError("Scenario argument must be JSON or 'Name:::Description'.")


def _load_scenario_from_path(path_value: str):
    path = _resolve_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and "name" in data and "description" in data:
        return Scenario.from_dict(data)
    if isinstance(data, dict) and ("l1_events" in data or "l2_events" in data):
        # Looks like a saved log file; not a scenario definition.
        return None
    raise ValueError(f"{path} does not look like a scenario JSON definition.")


def _load_l1_log_from_session_path(path_value: str) -> L1EventLog:
    path = _resolve_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Session directory not found: {path}")
    return load_l1_log_from_session(str(path))


def _load_l2_batch_from_source(path_value: str, prefer_sanitized: bool = False):
    """
    Load Layer 2 events from either a session directory or a standalone log file.
    Returns (payload, events, generated_at, is_sanitized, session_dir, source_path).
    """
    path = _resolve_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"L2 artifacts not found: {path}")

    payload = {}
    events_raw = []
    session_dir = None
    source_path = path
    is_sanitized = False
    timestamp_raw = None

    if path.is_dir():
        session_dir = path
        sanitized_path = path / "layer2_sanitized.json"
        raw_path = path / "layer2_raw.json"
        candidates = [sanitized_path, raw_path] if prefer_sanitized else [raw_path, sanitized_path]
        target = next((candidate for candidate in candidates if candidate.exists()), None)
        if not target:
            raise FileNotFoundError(f"No Layer 2 artifacts found in session directory: {path}")
        source_path = target
        is_sanitized = target.name == "layer2_sanitized.json"
        with open(target, "r", encoding="utf-8") as f:
            payload = json.load(f)
        events_raw = payload.get("events", [])
        timestamp_raw = payload.get("generated_at")
    else:
        payload = load_l2_event_log(str(path))
        events_raw = payload.get("l2_events", payload.get("events", []))
        timestamp_raw = payload.get("timestamp_created")
        is_sanitized = payload.get("sanitized", payload.get("llm_model") is None)

    events = []
    for evt in events_raw:
        try:
            parsed = deserialize_event(evt)
        except Exception as exc:  # pragma: no cover - defensive parse
            print(f"Failed to parse L2 event: {exc}")
            parsed = None
        if parsed:
            events.append(parsed)

    if not timestamp_raw:
        timestamp_raw = datetime.now().isoformat()
    try:
        generated_at = datetime.fromisoformat(timestamp_raw)
    except ValueError:
        generated_at = datetime.now()

    return payload, events, generated_at, is_sanitized, session_dir, source_path


def _resolve_cli_scenario(scenario_arg: str = None, scenario_path: str = None):
    """Load or create a scenario based on CLI arguments."""
    if scenario_arg:
        scenario = _scenario_from_argument(scenario_arg)
        target_path = _resolve_path(scenario_path) if scenario_path else _default_scenario_path(scenario.name)
        saved_path = save_scenario_to_file(scenario, str(target_path))
        print(f"Saved scenario JSON to {saved_path}")
        return scenario

    if scenario_path:
        try:
            scenario = _load_scenario_from_path(scenario_path)
            if scenario:
                print(f"Loaded scenario from {scenario_path}")
                return scenario
        except FileNotFoundError as exc:
            print(exc)
        except ValueError as exc:
            print(exc)

    return None


def _summarize_layer2_events(events):
    total = len(events)
    behavioral = [e for e in events if hasattr(e, "category")]
    interaction = [e for e in events if hasattr(e, "initiator")]
    context = [e for e in events if hasattr(e, "activity_type")]
    print(f"Total L2 events: {total}")
    print(f"  Behavioral events: {len(behavioral)}")
    print(f"  Interaction events: {len(interaction)}")
    print(f"  Context events: {len(context)}")

def test_end_to_end_reconstruction(model: str = None, lan: str = "en", scenario: Scenario = None):
    """
    Full pipeline test:
    Scenario → L1 events (saved) → L2 events (reconstructed) → Scenario (reconstructed)
    """
    print("\n" + "="*80)
    print("SCENARIO RECONSTRUCTION PIPELINE TEST")
    print("="*80)
    
    # Step 1: Create or load a test scenario
    print("\n[1/4] Creating test scenario...")
    scenario = scenario or Scenario(
        name="Test_Reconstruction",
        description="A small group of children are engaged in free play with blocks. "
                    "One child becomes upset when another takes a block, leading to a "
                    "brief conflict resolved by an adult.",
        duration_minutes=3,
        num_children=3,
        num_adults=1,
    )
    print(f"  Scenario: {scenario.name}")
    print(f"  Description: {scenario.description[:60]}...")
    
    # Step 2: Generate Layer 1 events and persist session artifacts
    print("\n[2/4] Generating Layer 1 events...")
    engine = ObservationEngine(model_id=model, artifact_lan=lan)
    handle = engine.create_session(scenario)
    layer1_batch = engine.run_layer1(handle)
    l1_events = layer1_batch.events
    print(f"  Generated {len(l1_events)} Layer 1 events")
    
    # Create and save L1 log
    log_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    duration_seconds = int((l1_events[-1].timestamp - l1_events[0].timestamp).total_seconds()) if l1_events else 0
    
    l1_log = L1EventLog(
        log_id=log_id,
        scenario_name=scenario.name,
        scenario_description=scenario.description,
        timestamp_created=datetime.now(),
        duration_seconds=duration_seconds,
        num_children=scenario.num_children,
        num_adults=scenario.num_adults,
        l1_events=l1_events,
    )
    
    session_dir = SESSIONS_DIR / handle.session_id
    print(f"  Session artifacts stored in: {session_dir}")
    
    # Step 3: Run ObservationEngine inference pipeline
    print("\n[3/4] Running ObservationEngine inference pipeline...")
    engine_layer2_batch = engine.run_layer2(handle)
    print(f"  Engine produced {len(engine_layer2_batch.events)} Layer 2 events")
    sanitized_batch = engine.sanitize_layer2(handle)
    print(f"  Sanitized Layer 2 events: {len(sanitized_batch.events)}")
    
    # Step 4: Reconstruct scenario from Layer 2 events
    print("\n[4/4] Reconstructing scenario from Layer 2 events...")    
    recon_artifact = engine.reconstruct(handle)
    print("  Engine reconstruction ready")
    
    # Display results
    print("\n" + "="*80)
    print("RECONSTRUCTION RESULTS")
    print("="*80)
    
    print("\n[ORIGINAL SCENARIO]")
    print(f"Name: {scenario.name}")
    print(f"Description:\n{scenario.description}")
    
    print("\n[LAYER 1 SUMMARY]")
    print(f"Total L1 events: {len(l1_events)}")
    speech_events = [e for e in l1_events if hasattr(e, 'speaker')]
    proximity_events = [e for e in l1_events if hasattr(e, 'proximity_level')]
    gaze_events = [e for e in l1_events if hasattr(e, 'direction') and not hasattr(e, 'object_type')]
    object_events = [e for e in l1_events if hasattr(e, 'object_type')]
    ambient_events = [e for e in l1_events if hasattr(e, 'sound_type')]
    posture_events = [e for e in l1_events if hasattr(e, 'posture')]
    
    print(f"  Speech events: {len(speech_events)}")
    print(f"  Proximity events: {len(proximity_events)}")
    print(f"  Gaze events: {len(gaze_events)}")
    print(f"  Object events: {len(object_events)}")
    print(f"  Ambient audio events: {len(ambient_events)}")
    print(f"  Posture events: {len(posture_events)}")
    
    print("\n[LAYER 2 SUMMARY - LIVE RUN]")
    _summarize_layer2_events(engine_layer2_batch.events)
    
    print("\n[ENGINE RECONSTRUCTION OUTPUT]")
    print(recon_artifact.text)
    
    print("\n" + "="*80)
    print("RECONSTRUCTION PIPELINE TEST COMPLETE")
    print("="*80)
    
    return {
        "original_scenario": scenario,
        "l1_events": l1_events,
        "session_id": handle.session_id,
        "engine_layer2_events": engine_layer2_batch.events,
        "engine_sanitized_events": sanitized_batch.events,
        "engine_reconstruction": recon_artifact.text,
    }


def test_load_and_reconstruct():
    """
    Test loading a previously saved L1 log and reconstructing from it.
    """
    print("\n" + "="*80)
    print("TEST: LOAD AND RECONSTRUCT FROM SAVED L1 LOG")
    print("="*80)
    
    sessions_dir = SESSIONS_DIR
    session_dirs = sorted(
        [p for p in sessions_dir.iterdir() if (p / "layer1_raw.json").exists()]
    ) if sessions_dir.exists() else []

    if not session_dirs:
        print("No session artifacts with Layer 1 data found. Run test_end_to_end_reconstruction() first.")
        return None
    
    # Load the most recent one
    session_path = session_dirs[-1]
    print(f"Loading session artifacts from: {session_path.name}")
    
    l1_log = load_l1_log_from_session(str(session_path))
    print(f"  Scenario: {l1_log.scenario_name}")
    print(f"  L1 events: {len(l1_log.l1_events)}")
    
    # Run reconstruction pipeline using ObservationEngine replay
    print("\nRunning ObservationEngine replay pipeline...")
    engine = ObservationEngine()

    result = engine.reconstruct_from_l1_log(l1_log)
    
    print("\n[RECONSTRUCTION SUMMARY]")
    print(f"L2 events generated: {len(result['layer2_batch'].events)}")
    print(f"\nScenario reconstruction:\n{result['reconstruction'].text[:500]}...")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test scenario reconstruction pipeline")
    parser.add_argument(
        "--stage",
        choices=["l1", "l2", "l2-sanitizer", "reconstruct", "full", "all"],
        default="full",
        help="Test stage: l1, l2, l2-sanitizer, reconstruct, full, all",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Specific model to use (optional)",
    )
    parser.add_argument(
        "--lan",
        default="en",
        help="Language/artifact lan to pass to ObservationEngine (default: en)",
    )
    parser.add_argument(
        "--event_path",
        default=None,
        help=(
            "Path to a session directory (preferred) or an L2 log JSON file for "
            "the 'l2', 'l2-sanitizer', and 'reconstruct' stages."
        ),
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help=f"""Save new scenario name and decription to JSON file.
        """,
    )
    parser.add_argument(
        "--scn_path",
        default=None,
        help=(
            "Path to scenario JSON (for 'l1'/'full') or session directory when "
            "combined with the 'l2' stage."
        ),
    )
    
    args = parser.parse_args()
    stage = args.stage.lower()
    needs_scenario = bool(args.scenario)
    if not needs_scenario:
        if stage in {"l1", "full", "all"}:
            needs_scenario = True
        elif stage == "l2" and not args.event_path:
            needs_scenario = True

    scenario = _resolve_cli_scenario(args.scenario, args.scn_path) if needs_scenario else None

    if args.model:
        Config.load()  # Ensure config is loaded
        available_models = get_available_models()
        if args.model not in available_models.values():
            print("Select a model to test:\n" +
                "\n".join([f"{i+1}. {name} ({model_id})" for i, (name, model_id) in enumerate(available_models.items())]) +
                "\nEnter the number of the model to test: ")
            choice = int(input().strip()) - 1
            args.model = list(available_models.items())[choice]

    if stage == "l1":
        test_l1_event_generation(model=args.model, lan=args.lan, scenario=scenario)

    elif stage == "l2":
        # Load existing session artifacts or generate a new one
        l1_source_path = args.event_path or (args.scn_path if scenario is None else None)
        if l1_source_path:
            try:
                session_path = _resolve_path(l1_source_path)
                l1_log = load_l1_log_from_session(str(session_path))
                print(f"Loaded session artifacts from {session_path}")
                print(f"Scenario: {l1_log.scenario_name}")
                print(f"L1 events: {len(l1_log.l1_events)}")
            except FileNotFoundError as exc:
                print(exc)
                sys.exit(1)
        else:
            l1_log, session_id = test_l1_event_generation(model=args.model, lan=args.lan, scenario=scenario)
            session_path = SESSIONS_DIR / session_id

        engine = ObservationEngine(artifact_lan=args.lan)
        handle = engine.bootstrap_from_l1_log(l1_log)
        l2_batch = engine.run_layer2(handle)
        print("Layer2Composer output summary:")
        _summarize_layer2_events(l2_batch.events)

        print(f"Session directory: {session_path}")
        print("Layer 2 artifacts stored in session directory.")
        print("Run with --stage l2-sanitizer --event_path <session-path> to validate sanitization.")

    elif stage in {"l2-sanitizer", "l2-sanitiser"}:
        if not args.event_path:
            print("--event_path must point to a session directory or L2 log for l2-sanitizer stage.")
            sys.exit(1)

        try:
            payload, l2_events, generated_at, is_sanitized, session_dir, source_path = _load_l2_batch_from_source(
                args.event_path,
                prefer_sanitized=False,
            )
        except FileNotFoundError as exc:
            print(exc)
            sys.exit(1)

        if is_sanitized:
            print("Input already sanitized; re-validating events.")
        else:
            print(f"Loaded raw Layer 2 events from {source_path}")
        batch = Layer2Batch(events=l2_events, generated_at=generated_at, sanitized=is_sanitized)

        sanitizer = Layer2Sanitizer()
        sanitized_batch = sanitizer.sanitize(batch)
        print("Layer2Sanitizer output summary:")
        _summarize_layer2_events(sanitized_batch.events)

        if session_dir:
            sanitized_path = session_dir / "layer2_sanitized.json"
            sanitized_payload = {
                "generated_at": sanitized_batch.generated_at.isoformat(),
                "events": [serialize_event(evt) for evt in sanitized_batch.events],
            }
            with open(sanitized_path, "w", encoding="utf-8") as f:
                json.dump(sanitized_payload, f, ensure_ascii=False, indent=2)
            print(f"Saved sanitized Layer 2 events to {sanitized_path}")
        else:
            print("Sanitized events ready (session directory not provided; results not saved).")

    elif stage == "reconstruct":
        if not args.event_path:
            print("--event_path must point to a session directory or L2 log JSON file for reconstruct stage.")
            sys.exit(1)

        try:
            payload, l2_events, generated_at, is_sanitized, session_dir, source_path = _load_l2_batch_from_source(
                args.event_path,
                prefer_sanitized=True,
            )
            print(f"Loaded {len(l2_events)} L2 events from {source_path}")
        except FileNotFoundError as exc:
            print(exc)
            sys.exit(1)

        l2_batch = Layer2Batch(events=l2_events, generated_at=generated_at, sanitized=is_sanitized)

        engine = ObservationEngine(artifact_lan=args.lan)
        scenario_name = payload.get("l1_log_id") or (session_dir.name if session_dir else "unknown")
        scenario = Scenario(
            name=f"Replay_{scenario_name}",
            description="Reconstruction replay from stored Layer 2 events.",
        )
        handle = engine.create_session(scenario, source="replay")
        if is_sanitized:
            handle.attach_layer2_sanitized(l2_batch)
        else:
            handle.attach_layer2_raw(l2_batch)

        artifact = engine.reconstruct(handle)
        print(f"Reconstruction for L1 log {scenario_name}:")
        print(artifact.text)

    elif stage in {"full", "all"}:
        result = test_end_to_end_reconstruction(model=args.model, lan=args.lan, scenario=scenario)

        if stage == "all":
            _ = test_load_and_reconstruct()
