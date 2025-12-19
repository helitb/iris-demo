"""
Session storage helpers for the observation engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .llm import get_config
from .storage import serialize_event
from .session import SessionHandle
from .paths import resolve_sessions_directory


class SessionStore:
    """Persists session artifacts under a single directory."""

    def __init__(self, base_directory: Optional[str] = None):
        config = get_config()
        target_directory = base_directory or config.sessions_directory
        self.base_directory = resolve_sessions_directory(target_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        session_dir = self.base_directory / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def persist(self, handle: SessionHandle) -> Path:
        """Persist all available artifacts for the given handle."""
        session_dir = self._session_dir(handle.session_id)
        artifacts = handle.artifacts

        # Scenario context
        scenario_path = session_dir / "scenario.json"
        with open(scenario_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": artifacts.scenario.name,
                    "description": artifacts.scenario.description,
                    "duration_minutes": artifacts.scenario.duration_minutes,
                    "num_children": artifacts.scenario.num_children,
                    "num_adults": artifacts.scenario.num_adults,
                    "focus_children": artifacts.scenario.focus_children,
                    "key_moments": artifacts.scenario.key_moments,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        if artifacts.layer1_raw:
            path = session_dir / "layer1_raw.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "started_at": artifacts.layer1_raw.started_at.isoformat(),
                        "ended_at": artifacts.layer1_raw.ended_at.isoformat(),
                        "events": [serialize_event(evt) for evt in artifacts.layer1_raw.events],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        if artifacts.layer2_raw:
            path = session_dir / "layer2_raw.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "generated_at": artifacts.layer2_raw.generated_at.isoformat(),
                        "events": [serialize_event(evt) for evt in artifacts.layer2_raw.events],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        if artifacts.layer2_sanitized:
            path = session_dir / "layer2_sanitized.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "generated_at": artifacts.layer2_sanitized.generated_at.isoformat(),
                        "events": [serialize_event(evt) for evt in artifacts.layer2_sanitized.events],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        if artifacts.reconstruction:
            path = session_dir / "reconstruction.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write(artifacts.reconstruction.text.strip() + "\n")

        if artifacts.slp_report:
            path = session_dir / "slp_report.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write(artifacts.slp_report.text.strip() + "\n")

        if artifacts.social_story:
            path = session_dir / "social_story.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write(artifacts.social_story.text.strip() + "\n")

        metadata_path = session_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_id": handle.session_id,
                    "state": handle.state,
                    "models": handle.models,
                    "created_at": handle.created_at.isoformat(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return session_dir
