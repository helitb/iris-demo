# Implementation Notes (Backend Refresh)

This document maps the source files that make up the reconstruction backend and explains
how they fit together. Use it as a reference when editing prompts, swapping models, or
adding new layers.

## src/core/engine.py – `ObservationEngine`

- Orchestrates every stage.
- Injects dependencies (`LLMLayer1Source`, `Layer2Composer`, `Layer2Sanitizer`,
  `ScenarioReconstructor`, `SessionStore`).
- `create_session` builds a `SessionHandle` with scenario context.
- `run_layer1` streams events from the configured `Layer1Source`.
- `run_layer2` composes Layer 2 events using the stored Layer 1 batch.
- `sanitize_layer2` enforces privacy and persists the sanitized copy.
- `reconstruct` generates the narrative artifact and records which model did so.
- `bootstrap_from_l1_log` + `reconstruct_from_l1_log` allow replaying saved sessions.

## src/core/layer1.py – Layer 1 sources & logs

- `Layer1Source` is the abstract interface for anything that can produce Layer 1 events.
- `LLMLayer1Source` streams structured events from the LLM using the prompts defined in
  `src/core/prompting.py`.
- `L1EventLog` stores metadata and serialized `Layer1Event` objects; used for persistence
  and replay.
- `load_l1_log_from_session(session_dir)` reconstructs `L1EventLog` objects from files
  written by `SessionStore`.

## src/core/layer2.py – Composition & sanitization

- `Layer2Composer` converts Layer 1 events into higher-level behaviors/interactions using
  the `LAYER2_SYSTEM_PROMPT`.
- `Layer2Sanitizer` runs each output through `validate_layer2_no_pii` (imported from
  `src/core/privacy.py`). It prints warnings and drops offending events, returning a new
  `Layer2Batch` flagged as sanitized.

## src/core/reconstructor.py – Narrative generation

- `_format_l2_events_for_llm` converts sanitized events into readable bullet points.
- `ScenarioReconstructor.reconstruct` switches between English/Hebrew instructions based on
  the engine’s `artifact_lan` parameter and generates a `ReconstructionArtifact`.

## src/core/privacy.py – Validation utilities

- `validate_layer2_no_pii` contains the guardrails that detect quoted speech or speech
  attribution verbs.
- `validate_l2_events` builds a `PrivacyValidationReport` (success rate + failed IDs) when
  you want a structured summary.
- `anonymized_session` and `layer1_only_session` are helpers used when exporting data for
  review outside the secure boundary.

## src/core/session.py – Dataclasses

- `Layer1Batch`, `Layer2Batch`, `ReconstructionArtifact`, `SessionArtifacts`, and
  `SessionHandle` keep track of intermediate state and model IDs.
- Every transition updates the `state` field (`layer1_raw`, `layer2_raw`, `layer2_sanitized`,
  `reconstructed`) so persistence logic knows what to write.

## src/core/session_store.py – Persistence

- Writes JSON snapshots of every available artifact into `sessions/<session_id>/`.
- Always stores scenario metadata and handle metadata, even if certain layers have not been
  executed yet.
- Creates directories on demand based on `config.sessions_directory`.

## src/core/llm.py – Client + config

- `Config.load` reads `config.yaml` and exposes generation/token settings.
- `LLMClient` abstracts whether a model is called via Anthropic’s SDK or aisuite and
  exposes `stream_and_parse`/`complete`.
- `get_available_models` automatically hides aisuite-only identifiers when aisuite is not
  installed.

## src/core/prompting.py – Prompts & parsers

- Contains both Layer 1 and Layer 2 system prompts plus `parse_event_line` helpers.
- Central place to tweak instructions when the sanitizer starts dropping too many events.

## src/tests/test_reconstruction.py – CLI harness

- Provides the `--stage` switch used throughout the docs.
- Uses `ObservationEngine` exclusively—no hidden helper classes—so it reflects how the
  backend should be consumed.
- Doubles as a regression test when run under `pytest` or directly via `python`.

## Key behavioral contracts

1. **Always sanitize Layer 2 before reconstruction.** `ObservationEngine.reconstruct`
   prefers sanitized events but will fall back to raw ones if necessary. Avoid that path in
   production code.
2. **Avoid direct access to the LLM client.** Let `ObservationEngine` own the `LLMClient` so
   swapping models (`ObservationEngine.swap_model`) updates every dependent component.
3. **Persist after each stage.** `ObservationEngine` calls `SessionStore.persist` after every
   layer—follow that pattern when extending the engine so replay tools keep working.

## Future work anchors

- When adding metrics, consider storing them alongside `metadata.json` inside each session
  directory so replay tools can surface them.
- To integrate real sensor data, implement a new `Layer1Source` subclass that reads from
  recorded files and pass it into `ObservationEngine(layer1_source=MySource(...))`.
- The Streamlit app will eventually wrap these backend calls again; keep the API surface of
  `ObservationEngine` stable to ease that integration.
