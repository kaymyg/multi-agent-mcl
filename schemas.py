"""
schemas.py
----------
Data Contract & Structured Output Layouts for the
Multi-Agent Meta-Cognitive Calibration Layer (MCL).

Dependencies: pydantic
"""

from typing import List, Literal
from pydantic import BaseModel, Field


class ProbeSchema(BaseModel):
    probe_id: str = Field(description="Unique system tracker string, e.g., probe_1049")
    scenario_prompt: str = Field(description="Adversarial prompt designed to challenge agent policies.")
    evaluation_invariant: str = Field(description="The deterministic constraint logic that must be satisfied.")
    category: Literal["edge_case", "multi_turn", "heuristic_failure"]


class CanarySuiteSchema(BaseModel):
    probes: List[ProbeSchema]
