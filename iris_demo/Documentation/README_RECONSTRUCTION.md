# IRIS Backend Documentation Index

The IRIS demo now exposes a single backend pipeline built around the `ObservationEngine`
(`src/core/engine.py`). This document is the jumping-off point for everything related to
Layer 1 generation, Layer 2 composition, privacy enforcement, and scenario reconstruction.

> ⚠️ The Streamlit front-end (`app.py`) has not been refreshed yet. Treat the CLI and the
> backend modules as the source of truth until the UI work lands.

## 📚 Documentation Map

| Need | Where to go |
|------|-------------|
| End-to-end instructions & commands | [RECONSTRUCTION_QUICKSTART.md](./RECONSTRUCTION_QUICKSTART.md) |
| Deep architecture & privacy notes | [RECONSTRUCTION.md](./RECONSTRUCTION.md) |
| Module-by-module breakdown | [IMPLEMENTATION.md](./IMPLEMENTATION.md) |

## 🧠 What the backend does

```
Scenario → Layer 1 events (LLMLayer1Source)
        → Layer 2 events (Layer2Composer)
        → Privacy check (Layer2Sanitizer / validate_layer2_no_pii)
        → Narrative artifact (ScenarioReconstructor)
```

Everything is orchestrated from `ObservationEngine`, which manages storage via
`SessionStore` and keeps track of the model used at each stage. Saved sessions live under
`sessions/` and always include:

```
sessions/<session_id>/
├── scenario.json
├── layer1_raw.json
├── layer2_raw.json
├── layer2_sanitized.json   # only after sanitize()
├── reconstruction.txt      # only after reconstruct()
└── metadata.json
```

## 🚀 Fast path

```bash
cd iris_demo
python src/tests/test_reconstruction.py --stage full --model claude-sonnet-4-20250514
```

The script bootstraps a scenario, runs the full pipeline, and prints the reconstruction.
Use `--stage l1`, `--stage l2`, `--stage l2-sanitizer`, or `--stage reconstruct` to run
individual steps. See the quickstart guide for details.

## 🔑 Configuration & prerequisites

- API keys: set `ANTHROPIC_API_KEY` in your shell or `.env`. (aisuite models require the
  matching provider keys as well.)
- Model choices live in `config.yaml`. Use its `models` section to add new IDs.
- Dependencies: install `requirements.txt` in a virtualenv.

## 🗂️ Code map

```
src/
├── core/
│   ├── engine.py          # ObservationEngine orchestrator
│   ├── layer1.py          # L1 sources and L1EventLog helpers
│   ├── layer2.py          # Layer2Composer + Layer2Sanitizer
│   ├── reconstructor.py   # ScenarioReconstructor (L2 → narrative)
│   ├── privacy.py         # PrivacyValidationReport utilities
│   ├── session.py         # Artifact dataclasses
│   ├── session_store.py   # Disk persistence for artifacts
│   ├── llm.py             # LLM client + config loader
│   └── prompting.py       # Prompt templates & parsers
├── tests/
│   └── test_reconstruction.py   # CLI harness & regression test
└── __init__.py            # Package exports (`from src import ObservationEngine`, …)
```

## 🧪 Recommended workflow (backend only)

1. **Configure models** in `config.yaml` and load env vars.
2. **Run `python src/tests/test_reconstruction.py --stage full`** to confirm the pipeline.
3. **Inspect `sessions/<session_id>/`** to review raw artifacts.
4. **Iterate on prompts/policies** inside `src/core/prompting.py`, `layer2.py`, or
   `reconstructor.py`.
5. **Rerun the targeted stage** with `--stage l2` or `--stage reconstruct` for fast loops.

## 🧭 Support / FAQ

- **Need an example script?** See `RECONSTRUCTION_QUICKSTART.md`.
- **Want to understand each class?** `IMPLEMENTATION.md` documents the backend surface.
- **Why is there no `EventGenerator` anymore?** Layer 1 is produced through
  `LLMLayer1Source`, which lives inside `ObservationEngine`—all previous generator API
  calls should now go through the engine.
- **Where is privacy handled?** `Layer2Sanitizer` (in `layer2.py`) + `validate_layer2_no_pii`
  (in `privacy.py`) run before any reconstruction step, and the sanitization results are
  persisted under each session directory.

---

_Last updated: December 2025 (backend refresh)._
