# Reconstruction Pipeline – Quickstart (Backend Only)

Use this guide when you need to exercise the backend without touching the out-of-date
Streamlit UI.

## 1. Prerequisites

1. Install dependencies: `pip install -r requirements.txt`.
2. Provide API keys:
   - `ANTHROPIC_API_KEY` for direct Claude models (preferred path used by default).
   - For aisuite models add the provider-specific key(s) to your environment or `.env`.
3. Review `config.yaml` to confirm the default model and directory paths.

## 2. One-command sanity check

```bash
cd iris_demo
python src/tests/test_reconstruction.py --stage full
```

What you get:
- Generates a sample scenario.
- Runs Layer 1 generation (`ObservationEngine.run_layer1`).
- Composes Layer 2 (`run_layer2`) and sanitizes it (`sanitize_layer2`).
- Produces a narrative via `ObservationEngine.reconstruct`.
- Persists everything to `sessions/<session_id>/`.

Change `--stage` to target specific pieces:

| Stage | What it does |
|-------|--------------|
| `l1` | Only generates Layer 1 events and saves them. |
| `l2` | Replays the latest L1 log (or the one passed with `--event_path`) and produces unsanitized L2. |
| `l2-sanitizer` | Loads existing L2 payloads (`--event_path`) and re-runs the sanitizer. |
| `reconstruct` | Loads L2 events (sanitized preferred) and regenerates the narrative only. |
| `full` | End-to-end single pass (default). |
| `all` | Runs `full` followed by the replay-only test. |

Additional useful flags:

- `--model <model_id>` to override the default from `config.yaml`.
- `--lan <en|he>` to control the reconstruction language (propagates to `ScenarioReconstructor`).
- `--event_path <session_dir_or_log>` when replaying existing artifacts.

## 3. Minimal Python usage

### End-to-end run

```python
from datetime import datetime
from src import ObservationEngine, Scenario, L1EventLog

engine = ObservationEngine(model_id="claude-sonnet-4-20250514")

scenario = Scenario(
    name="Peer Breakthrough",
    description="Two children collaborate on circle time props...",
    duration_minutes=3,
    num_children=3,
    num_adults=1,
)

handle = engine.create_session(scenario)
layer1 = engine.run_layer1(handle)
layer2 = engine.run_layer2(handle)
sanitized = engine.sanitize_layer2(handle)
artifact = engine.reconstruct(handle)

print(f"Layer 1 events: {len(layer1.events)}")
print(f"Layer 2 events (sanitized): {len(sanitized.events)}")
print("\nReconstruction:\n", artifact.text)
```

### Replay from a saved L1 log

```python
from src import load_l1_log_from_session, ObservationEngine

l1_log = load_l1_log_from_session("sessions/<session_id>")
engine = ObservationEngine()
result = engine.reconstruct_from_l1_log(l1_log)

print(f"L2 events: {len(result['sanitized_layer2_batch'].events)}")
print(result["reconstruction"].text[:500])
```

### Creating an `L1EventLog` manually

```python
from src import L1EventLog
from datetime import datetime

l1_log = L1EventLog(
    log_id="custom_001",
    scenario_name=scenario.name,
    scenario_description=scenario.description,
    timestamp_created=datetime.now(),
    duration_seconds=layer1.ended_at and int((layer1.ended_at - layer1.started_at).total_seconds()) or 0,
    num_children=scenario.num_children,
    num_adults=scenario.num_adults,
    llm_model=engine.llm_client.model,
    l1_events=layer1.events,
)
```

Save/load logic is handled automatically by `SessionStore`, but the dataclass is available
when you need to serialize custom batches.

## 4. Working with session artifacts

- All runs write to `sessions/<session_id>/`.
- `layer1_raw.json` / `layer2_raw.json` store the serialized events.
- `layer2_sanitized.json` only exists after calling `sanitize_layer2`.
- `reconstruction.txt` only exists after `reconstruct`.
- Metadata includes the models used at each stage for traceability.

You can hand-edit or copy any of these JSON artifacts and feed them back through the CLI
with `--event_path`.

## 5. Troubleshooting checklist

| Problem | Likely cause / fix |
|---------|-------------------|
| `anthropic` errors | Missing `ANTHROPIC_API_KEY`. Export it or add to `.env`. |
| aisuite models unavailable | `pip install aisuite[openai,anthropic]` (already in requirements) and use Python 3.10+. |
| `Layer 1 batch missing` errors | You called `run_layer2`/`sanitize_layer2` before `run_layer1`. Re-run starting from Layer 1 or bootstrap with `load_l1_log_from_session`. |
| Sanitizer drops events | `validate_layer2_no_pii` flagged quoted speech. Update prompts so the LLM describes behavior instead of quoting. |
| Nothing under `sessions/` | Ensure `config.yaml` → `sessions.directory` points to a writable folder. The default `sessions/` in repo root should exist after the first run. |

## 6. Next steps

Once the quickstart succeeds:
1. Read `RECONSTRUCTION.md` for the architectural deep dive (Layer 1 → Layer 2 → sanitized
   narrative with privacy constraints).
2. Use `IMPLEMENTATION.md` as a reference when editing prompts or experimenting with new
   models.
3. If you need to add higher-level orchestration or integrate with the UI, keep in mind that
   the backend is the source of truth until the Streamlit layer is updated.
