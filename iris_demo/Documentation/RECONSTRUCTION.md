# IRIS Scenario Reconstruction – Architecture Overview

The current backend couples all layer transitions into a single orchestrator:
`ObservationEngine` (`src/core/engine.py`). The sections below describe how each stage
works, how artifacts are stored, and where privacy is enforced.

## 1. Pipeline stages

| Stage | Entry point | Output | Notes |
|-------|-------------|--------|-------|
| 1. Layer 1 generation | `ObservationEngine.run_layer1` → `LLMLayer1Source` (`src/core/layer1.py`) | `Layer1Batch` (events + timestamps) | Generates ~40–80 low-level events via the configured LLM model. |
| 2. Layer 2 composition | `ObservationEngine.run_layer2` → `Layer2Composer` (`src/core/layer2.py`) | `Layer2Batch` (unsanitized) | Consumes saved Layer 1 events only. No scenario description is passed to the LLM. |
| 3. Sanitization / privacy | `ObservationEngine.sanitize_layer2` → `Layer2Sanitizer` + `validate_layer2_no_pii` (`src/core/privacy.py`) | `Layer2Batch` (sanitized copy) | Drops any event whose description hints at PII (quoted speech, verbs like “said”). |
| 4. Scenario reconstruction | `ObservationEngine.reconstruct` → `ScenarioReconstructor` (`src/core/reconstructor.py`) | `ReconstructionArtifact` (narrative text) | Receives sanitized Layer 2 events, not the original scenario. Language is configurable via `artifact_lan`. |

Every call stores artifacts in `SessionStore` (`src/core/session_store.py`). Each session
knows which model produced each layer via `SessionHandle.models`.

## 2. Data flow

```
ScenarioContext (scenario + source metadata)
       │
       ▼
Layer1Batch
  - timestamp range
  - ~40-80 events of types:
      SpeechEvent, AmbientAudioEvent,
      ProximityEvent, GazeEvent, PostureEvent, ObjectEvent
       │
       ▼
Layer2Batch (raw)
  - BehavioralEvent (~4-10)
  - InteractionEvent (~4-8)
  - ContextEvent (~2-4)
       │
       ▼
Layer2Batch (sanitized)
  - Only events that pass `validate_layer2_no_pii`
  - Deep-copied to avoid mutating the raw batch
       │
       ▼
ReconstructionArtifact
  - free-form text narrative in English or Hebrew
  - stored as `reconstruction.txt`
```

## 3. Privacy guarantees

1. **No scenario leakage** – `Layer2Composer` receives only serialized Layer 1 events
   retrieved from disk. The original scenario text is never part of the prompt.
2. **PII filter** – `validate_layer2_no_pii` inspects the description (and trigger
   description) fields in every Layer 2 event. It rejects anything containing quotes or
   speech-attribution verbs (“said”, “stated”, “uttered”, “reported”). Failed events are
   logged to stdout and omitted from the sanitized batch.
3. **Sanitized-only reconstruction** – `ObservationEngine.reconstruct` always prefers the
   sanitized batch (`layer2_sanitized`) when available. Passing raw events is only possible
   if you call `reconstruct` before `sanitize_layer2`, which should be avoided outside of
   debugging loops.
4. **Persistent audit trail** – Each session directory contains both raw and sanitized
   payloads so a reviewer can confirm what was removed and why.

## 4. Storage layout

```
sessions/<session_id>/
├── scenario.json                # copy of the Scenario dataclass
├── layer1_raw.json              # Layer1Batch serialized events
├── layer2_raw.json              # unsanitized Layer 2 events
├── layer2_sanitized.json        # created after sanitize_layer2()
├── reconstruction.txt           # narrative text
└── metadata.json                # stage → model map, timestamps, session state
```

Use `load_l1_log_from_session(session_dir)` to reconstruct an `L1EventLog` from the stored
JSON files. That method is what powers the replay paths in `test_reconstruction.py`.

## 5. Configuration knobs

| Setting | Location | Description |
|---------|----------|-------------|
| Default model + alternatives | `config.yaml` | Each entry maps a friendly label to the actual model ID; colon-prefixed IDs go through aisuite. |
| Token budgets | `config.yaml → generation` | `layer1_max_tokens`, `layer2_max_tokens`, `report_max_tokens` feed directly into the `LLMClient`. |
| Session/report directories | `config.yaml → sessions / reports` | Change these if you need to store artifacts outside the repo. |
| Reconstruction language | `ObservationEngine(artifact_lan=...)` | `ScenarioReconstructor` switches prompts between English and Hebrew. |

## 6. Testing & observability

- `python src/tests/test_reconstruction.py --stage full`: the canonical regression test.
- `--stage l2` and `--stage l2-sanitizer` make it easy to inspect the intermediate
  artifacts without re-running Layer 1.
- Sanitization failures are printed to stdout with a helpful message that explains why the
  description was rejected.
- Each `SessionHandle` tracks its `state`, so you can tell from `metadata.json` whether a
  particular run stopped after Layer 2, after sanitization, or after reconstruction.

## 7. Extending the pipeline

Areas that are ready for further work:

1. **Quantitative evaluation** – add embeddings/ROUGE scoring modules that compare the
   sanitized narrative against the original scenario (store metrics alongside sessions).
2. **Advanced privacy reporting** – persist structured `PrivacyValidationReport` objects so
   the sanitizer outcome is machine-readable.
3. **Prompt experimentation** – prompts live in `src/core/prompting.py`; use the CLI stages
   to iterate quickly.
4. **Alternate L1 sources** – implement another `Layer1Source` subclass (e.g., loading
   sensor dumps) and pass it into `ObservationEngine(layer1_source=...)`.

Everything above can be developed without touching the Streamlit UI; once stable, wire it
into `app.py`.
