"""
Session handle and artifact containers for the ObservationEngine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .schema import Layer1Event, Layer2Event
from .scenario import Scenario
from .context import ScenarioContext


@dataclass
class Layer1Batch:
    """Collection of Layer 1 events and bookkeeping metadata."""

    events: List[Layer1Event]
    started_at: datetime
    ended_at: datetime
    raw_stream: Optional[str] = None


@dataclass
class Layer2Batch:
    """Collection of Layer 2 events."""

    events: List[Layer2Event]
    generated_at: datetime
    sanitized: bool = False


@dataclass
class ReconstructionArtifact:
    """Scenario reconstruction narrative."""

    text: str
    type: Optional[str] = "narrative"
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SessionArtifacts:
    """All artifacts associated with a session run."""

    scenario: Scenario
    layer1_raw: Optional[Layer1Batch] = None
    layer2_raw: Optional[Layer2Batch] = None
    layer2_sanitized: Optional[Layer2Batch] = None
    reconstruction: Optional[ReconstructionArtifact] = None
    slp_report: Optional[ReconstructionArtifact] = None
    social_story: Optional[ReconstructionArtifact] = None


@dataclass
class SessionHandle:
    """
    Tracks the lifecycle of a scenario run.

    The handle stores intermediate artifacts so they can be persisted after each
    step. It also maintains a shared session_id used to group files.
    """

    session_id: str
    context: ScenarioContext
    created_at: datetime = field(default_factory=datetime.now)
    models: Dict[str, str] = field(default_factory=dict)
    artifacts: SessionArtifacts = field(init=False)
    state: str = "initialized"

    def __post_init__(self) -> None:
        self.artifacts = SessionArtifacts(scenario=self.context.scenario)

    def set_model(self, stage: str, model_id: Optional[str]) -> None:
        if model_id:
            self.models[stage] = model_id

    def set_state(self, state: str) -> None:
        self.state = state

    def attach_layer1(self, batch: Layer1Batch) -> None:
        self.artifacts.layer1_raw = batch
        self.set_state("layer1_raw")

    def attach_layer2_raw(self, batch: Layer2Batch) -> None:
        self.artifacts.layer2_raw = batch
        self.set_state("layer2_raw")

    def attach_layer2_sanitized(self, batch: Layer2Batch) -> None:
        self.artifacts.layer2_sanitized = batch
        self.set_state("layer2_sanitized")

    def attach_reconstruction(self, artifact: ReconstructionArtifact) -> None:
        self.artifacts.reconstruction = artifact
        self.set_state("reconstructed")

    def attach_social_story(self, artifact: ReconstructionArtifact) -> None:
        self.artifacts.social_story = artifact
        self.artifacts.social_story.type = "social_story"
        self.set_state("social_story_generated")

    def attach_slp_report(self, artifact: ReconstructionArtifact) -> None:
        self.artifacts.slp_report = artifact
        self.artifacts.slp_report.type = "slp_report"
        self.set_state("slp_report_generated")