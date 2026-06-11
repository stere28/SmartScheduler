"""
SmartScheduler – Data Models
==============================
Pydantic models shared by all agents.
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ── Worker preference models ───────────────────────────────────────────────────

class ShiftPreference(BaseModel):
    """Structured representation of a single worker's scheduling preferences."""
    worker_id: str
    worker_name: str
    worker_type: Literal["standard", "specialized"] = "standard"

    # Preferred shifts (list of "morning" | "afternoon" | "night")
    preferred_shifts: list[str] = Field(default_factory=list)
    # Shifts the worker wants to avoid
    avoided_shifts: list[str]   = Field(default_factory=list)

    # Tolerances (True = accepts / False = wants to avoid)
    night_tolerance:   bool = True
    holiday_tolerance: bool = True

    # Availability: list of day indices (0=Mon, …, 6=Sun) the worker is NOT available
    unavailable_days_of_week: list[int] = Field(default_factory=list)

    # Preferred rest day of week (0=Mon … 6=Sun), None = no preference
    preferred_rest_day: Optional[int] = None

    # Emergency coverage per month (number of extra shifts the worker accepts)
    emergency_coverage: int = 0

    # Raw natural-language text provided by the worker
    raw_text: str = ""

    @field_validator("preferred_shifts", "avoided_shifts", "unavailable_days_of_week", mode="before")
    @classmethod
    def _coerce_none_to_list(cls, v):
        """Convert null/None from LLM output to an empty list."""
        return v if v is not None else []


class WorkforcePreferences(BaseModel):
    """Collection of all workers' preferences."""
    workers: list[ShiftPreference]


# ── Schedule models ────────────────────────────────────────────────────────────

class Assignment(BaseModel):
    """A single shift assignment: (worker_id, day_index, shift_index)."""
    worker_id:   str
    day_index:   int   # 0-based index into scheduling horizon
    shift_index: int   # 0=morning, 1=afternoon, 2=night

    @property
    def shift_name(self) -> str:
        from config import SHIFT_NAMES
        return SHIFT_NAMES[self.shift_index]


class Schedule(BaseModel):
    """A full schedule for the planning horizon."""
    assignments: list[Assignment] = Field(default_factory=list)

    def assignments_for_worker(self, worker_id: str) -> list[Assignment]:
        return [a for a in self.assignments if a.worker_id == worker_id]

    def assignments_for_day(self, day_index: int) -> list[Assignment]:
        return [a for a in self.assignments if a.day_index == day_index]

    def assignments_for_shift(self, day_index: int, shift_index: int) -> list[Assignment]:
        return [a for a in self.assignments
                if a.day_index == day_index and a.shift_index == shift_index]


# ── Verification models ────────────────────────────────────────────────────────

class ConstraintViolation(BaseModel):
    description: str
    severity: Literal["hard", "soft"] = "hard"


class VerificationReport(BaseModel):
    passed: bool
    violations: list[ConstraintViolation] = Field(default_factory=list)
    fairness_scores: dict[str, float]     = Field(default_factory=dict)
    min_satisfaction: float = 0.0
    most_disadvantaged_worker: Optional[str] = None


# ── Agent state (LangGraph) ────────────────────────────────────────────────────

class SmartSchedulerState(BaseModel):
    """Shared state propagated through the LangGraph pipeline."""
    use_case: str = "A"                      # "A" or "B"
    preferences: Optional[WorkforcePreferences] = None
    schedule:    Optional[Schedule]            = None
    verification: Optional[VerificationReport] = None
    iteration:   int = 0
    converged:   bool = False
    history:     list[str] = Field(default_factory=list)  # log of actions
    
class RefinementStrategy(BaseModel):
    reasoning: str = Field(description="Brief explanation of the strategy")
    shifts_to_avoid: list[int] = Field(description="List of shift indices (0=morning, 1=afternoon, 2=night) to strictly ban for this worker", default=[])
    shifts_to_prefer: list[int] = Field(description="List of shift indices to encourage", default=[])
    weight_boost: int = Field(description="An integer multiplier from 1 to 10 to boost this worker's objective weight", default=5)
