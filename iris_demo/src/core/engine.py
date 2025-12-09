"""
Observation engine orchestrating the IRIS pipeline.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, Dict

from .llm import LLMClient
from .scenario import Scenario
from .context import ScenarioContext
from .layer1 import Layer1Source, LLMLayer1Source, L1EventLog
from .layer2 import Layer2Composer, Layer2Sanitizer
from .session import SessionHandle, Layer1Batch
from .session_store import SessionStore
from .reconstructor import ScenarioReconstructor


class ObservationEngine:
    """Coordinated pipeline for Layer 1, Layer 2, and reconstruction phases."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        artifact_lan: Optional[str] = "en",
        store: Optional[SessionStore] = None,
        layer1_source: Optional[Layer1Source] = None,
    ):
        self.llm_client = LLMClient(model_id)
        self.layer1_source = layer1_source or LLMLayer1Source(self.llm_client)
        self.layer2_composer = Layer2Composer(self.llm_client)
        self.layer2_sanitizer = Layer2Sanitizer()
        self.reconstructor = ScenarioReconstructor(self.llm_client, language=artifact_lan or "en")
        self.store = store or SessionStore()

    def create_session(self, scenario: Scenario, source: str = "llm") -> SessionHandle:
        context = ScenarioContext(scenario=scenario, source=source)
        session_id = uuid.uuid4().hex[:12]
        return SessionHandle(session_id=session_id, context=context)

    def swap_model(self, model_id: str):
        """Swap the LLM model used in the engine and its components."""
        self.llm_client = LLMClient(model_id)
        self.layer1_source.llm_client = self.llm_client
        self.layer2_composer.llm_client = self.llm_client
        self.reconstructor.llm_client = self.llm_client
        
    # ------------------------------------------------------------------ Layer 1
    def run_layer1(
        self,
        handle: SessionHandle,
        on_event=None,
    ):
        batch = self.layer1_source.produce(handle.context.scenario, on_event=on_event)
        handle.attach_layer1(batch)
        handle.set_model("layer1_llm_model", self.llm_client.model)
        self.store.persist(handle)
        return batch

    # ------------------------------------------------------------------ Layer 2
    def run_layer2(self, handle: SessionHandle, on_event=None):
        if not handle.artifacts.layer1_raw:
            raise ValueError("Layer 1 batch missing. Run run_layer1() first.")

        batch = self.layer2_composer.compose(
            handle.artifacts.layer1_raw.events,
            on_event=on_event,
        )
        handle.attach_layer2_raw(batch)
        handle.set_model("layer2_llm_model", self.llm_client.model)
        self.store.persist(handle)
        return batch

    def sanitize_layer2(self, handle: SessionHandle):
        if not handle.artifacts.layer2_raw:
            raise ValueError("Layer 2 batch missing. Run run_layer2() first.")

        sanitized = self.layer2_sanitizer.sanitize(handle.artifacts.layer2_raw)
        handle.attach_layer2_sanitized(sanitized)
        self.store.persist(handle)
        return sanitized

    # ---------------------------------------------------------------- Reconstruction
    def reconstruct(self, handle: SessionHandle):
        l2_batch = handle.artifacts.layer2_sanitized or handle.artifacts.layer2_raw
        if not l2_batch:
            raise ValueError("Layer 2 batch missing. Provide sanitized or raw Layer 2 events.")

        artifact = self.reconstructor.reconstruct(l2_batch.events)
        handle.attach_reconstruction(artifact)
        handle.set_model("reconstruction_llm_model", self.llm_client.model)

        self.store.persist(handle)
        return artifact

    # ------------------------------------------------------------------ Storage
    def persist(self, handle: SessionHandle):
        """Force persistence of current artifacts."""
        return self.store.persist(handle)

    # ------------------------------------------------------------------ Replay / Logs
    def _build_layer1_batch(self, events) -> Layer1Batch:
        if events:
            started = events[0].timestamp
            ended = events[-1].timestamp
        else:
            started = ended = datetime.now()
        print(f"Reconstructed L1 batch with {len(events)} events.")
        return Layer1Batch(events=list(events), started_at=started, ended_at=ended)

    def bootstrap_from_l1_log(self, l1_log: L1EventLog) -> SessionHandle:
        """Create a session handle populated with Layer 1 events from a saved log."""
        scenario = Scenario(
            name=l1_log.scenario_name,
            description=l1_log.scenario_description,
            duration_minutes=max(1, l1_log.duration_seconds // 60 or 1),
            num_children=l1_log.num_children,
            num_adults=l1_log.num_adults,
        )
        context = ScenarioContext(scenario=scenario, source="replay")
        session_id = l1_log.log_id or uuid.uuid4().hex[:12]
        handle = SessionHandle(session_id=session_id, context=context)
        handle.attach_layer1(self._build_layer1_batch(l1_log.l1_events))
        if l1_log.llm_model:
            handle.set_model("layer1_llm_model", l1_log.llm_model)
        self.store.persist(handle)
        return handle

    def reconstruct_from_l1_log(
        self,
        l1_log: L1EventLog,
        on_l2_event=None,
    ) -> Dict[str, object]:
        """
        Run the L1→L2→reconstruction pipeline from a saved L1 log.
        Returns a dictionary with handle, batches, and reconstruction artifact.
        """
        handle = self.bootstrap_from_l1_log(l1_log)
        l2_batch = self.run_layer2(handle, on_event=on_l2_event)
        sanitized_batch = self.sanitize_layer2(handle)
        reconstruction = self.reconstruct(handle)
        return {
            "handle": handle,
            "layer2_batch": l2_batch,
            "sanitized_layer2_batch": sanitized_batch,
            "reconstruction": reconstruction,
        }
