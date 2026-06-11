"""
SmartScheduler – Data Models
==============================
Pydantic models shared by all agents.

Nuovi tipi aggiunti per l'architettura FASE 1a/1b:
  - ExtensionSpec: specifica tipizzata (NON codice) per un vincolo dinamico
  - ConstraintClassification: output del Constraint Classifier Agent
  - DynamicModeWarning: avviso formale emesso prima di entrare in Fase 1b
"""

from __future__ import annotations
from typing import Any, Literal, Optional
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


# ── Extension models (Fase 1b) ────────────────────────────────────────────────

class ExtensionSpec(BaseModel):
    """
    Specifica TIPIZZATA per un vincolo dinamico (Extension Pattern).

    IMPORTANTE: Questo oggetto NON contiene codice Python.
    Contiene solo il nome del pattern e i suoi parametri numerici/stringa.
    Il codice che implementa il vincolo risiede ESCLUSIVAMENTE in
    solver_extensions.py, scritto e testato da ingegneri umani.
    """
    extension_type: Literal[
        "max_consecutive_nights",
        "min_gap_hours_between_shifts",
        "pair_always_same_shift",
        "max_shifts_of_type_per_week",
        "worker_group_min_rest_days",
    ]
    parameters: dict[str, Any] = Field(default_factory=dict)

    # Tracciabilità: quale frase dell'utente ha generato questo vincolo
    source_text: str = ""


class DynamicModeWarning(BaseModel):
    """
    Avviso formale emesso dal sistema ogni volta che si entra in Fase 1b.
    Deve essere mostrato all'utente prima dell'esecuzione.
    """
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    unknown_constraints: list[str] = Field(default_factory=list)
    extensions_to_apply: list[str] = Field(default_factory=list)
    warning_message: str = (
        "⚠️ MODALITÀ DINAMICA ATTIVA: uno o più vincoli richiesti non sono presenti "
        "nel solver statico testato. Verranno applicati Extension Pattern pre-compilati. "
        "Il risultato non è coperto da unit test end-to-end. "
        "Verificare manualmente il calendario prodotto prima della distribuzione."
    )
    requires_acknowledgement: bool = True


class ConstraintClassification(BaseModel):
    """
    Output del Constraint Classifier Agent (Fase 1a / Fase 1b).

    - 'known'  → tutti i vincoli sono Casi Noti → solver statico, nessun avviso
    - 'mixed'  → alcuni vincoli sono nuovi ma mappabili su Extension Pattern
    - 'refused'→ vincolo richiesto non mappabile e non sicuro da implementare
    """
    classification: Literal["known", "mixed", "refused"]

    # Parametri per il solver statico (sempre presenti)
    known_params: dict[str, Any] = Field(default_factory=dict)

    # Extension da applicare (solo se classification == "mixed")
    extensions: list[ExtensionSpec] = Field(default_factory=list)

    # Vincoli che non è stato possibile mappare (motivo del 'refused' o dell'avviso)
    unmappable_constraints: list[str] = Field(default_factory=list)

    # Avviso formale (presente solo se classification != 'known')
    warning: Optional[DynamicModeWarning] = None

    # True se l'utente ha esplicitamente confermato di voler procedere
    # in modalità dinamica (future-proof per UI interattiva)
    user_acknowledged: bool = False


# ── Agent state (LangGraph) ────────────────────────────────────────────────────

class SmartSchedulerState(BaseModel):
    """Shared state propagated through the LangGraph pipeline."""
    use_case: str = "A"                      # "A" or "B"
    input_file: Optional[str] = None         # Path to txt file with preferences
    rules_file: Optional[str] = None         # Path to txt file with rules
    preferences: Optional[WorkforcePreferences] = None
    schedule:    Optional[Schedule]            = None
    verification: Optional[VerificationReport] = None
    iteration:   int = 0
    converged:   bool = False
    history:     list[str] = Field(default_factory=list)  # log of actions

    # ── Fase 1a/1b ──────────────────────────────────────────────────────────
    # Risultato della classificazione dei vincoli
    constraint_classification: Optional[ConstraintClassification] = None

    # True se il run corrente sta usando Extension Pattern (Fase 1b)
    dynamic_mode_active: bool = False

    # Messaggio di rifiuto (se classification == 'refused')
    refusal_reason: Optional[str] = None
