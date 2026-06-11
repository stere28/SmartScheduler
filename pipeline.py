"""
SmartScheduler – LangGraph Pipeline
======================================
Wires together all agents into a stateful graph with:
  - FASE 1a: Constraint Classifier (percorso statico vs dinamico)
  - FASE 1b: Dynamic Extension (caso eccezionale, con difese)
  - STAGE 2-4: Drafting → Verification → Refinement loop (invariati)

Grafo di esecuzione:
  preferences → constraint_classifier
                    ├── [known]  → drafting → verification ⇄ refinement
                    ├── [mixed]  → dynamic_extension → drafting → verification ⇄ refinement
                    └── [refused]→ END (con refusal_reason nello stato)
"""

from __future__ import annotations
from typing import Literal

from langgraph.graph import StateGraph, END

from models import SmartSchedulerState
from agents import (
    preferences_agent,
    constraint_classifier_agent,
    dynamic_extension_agent,
    drafting_agent,
    verification_agent,
    refinement_agent,
)
from config import MAX_REFINEMENT_ITERATIONS


# ── Node wrappers ──────────────────────────────────────────────────────────────

def _preferences_node(state: dict) -> dict:
    s = SmartSchedulerState(**state)
    result = preferences_agent(s)
    return result.model_dump()


def _constraint_classifier_node(state: dict) -> dict:
    s = SmartSchedulerState(**state)
    result = constraint_classifier_agent(s)
    return result.model_dump()


def _dynamic_extension_node(state: dict) -> dict:
    s = SmartSchedulerState(**state)
    result = dynamic_extension_agent(s)
    return result.model_dump()


def _drafting_node(state: dict) -> dict:
    s = SmartSchedulerState(**state)
    result = drafting_agent(s)
    return result.model_dump()


def _verification_node(state: dict) -> dict:
    s = SmartSchedulerState(**state)
    result = verification_agent(s)
    return result.model_dump()


def _refinement_node(state: dict) -> dict:
    s = SmartSchedulerState(**state)
    result = refinement_agent(s)
    return result.model_dump()


# ── Routing functions ──────────────────────────────────────────────────────────

def _route_after_classifier(
    state: dict,
) -> Literal["dynamic_extension", "drafting", "end"]:
    """
    FASE 1a / 1b routing:
      - 'known'   → drafting (percorso statico, nessun avviso)
      - 'mixed'   → dynamic_extension (percorso dinamico con validazione)
      - 'refused' → end (richiesta non gestibile, refusal_reason nello stato)
    """
    s = SmartSchedulerState(**state)
    clf = s.constraint_classification

    if clf is None:
        # Classificatore non eseguito (non dovrebbe accadere): fallback sicuro
        return "drafting"

    if clf.classification == "known":
        return "drafting"
    elif clf.classification == "mixed":
        return "dynamic_extension"
    else:  # refused
        reason = s.refusal_reason or "Classificazione 'refused': richiesta non gestibile."
        print(f"\n[Pipeline] RIFIUTO FINALE: {reason}")
        return "end"


def _route_after_dynamic_extension(
    state: dict,
) -> Literal["drafting", "end"]:
    """
    Dopo la Fase 1b:
      - Se la validazione è passata → drafting
      - Se la validazione è fallita (refusal_reason impostato) → end
    """
    s = SmartSchedulerState(**state)
    if s.refusal_reason:
        print(f"\n[Pipeline] RIFIUTO DOPO VALIDAZIONE: {s.refusal_reason}")
        return "end"
    return "drafting"


def _route_after_verification(state: dict) -> Literal["refinement", "end"]:
    """After verification: if passed and not converged, refine; else end."""
    s = SmartSchedulerState(**state)
    if s.verification is None:
        return "end"
    if not s.verification.passed:
        return "end"
    if s.converged or s.iteration >= MAX_REFINEMENT_ITERATIONS:
        return "end"
    return "refinement"


def _route_after_refinement(state: dict) -> Literal["verify", "end"]:
    """After refinement: re-verify, unless converged."""
    s = SmartSchedulerState(**state)
    if s.converged or s.iteration >= MAX_REFINEMENT_ITERATIONS:
        return "end"
    return "verify"


# ── Build graph ────────────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    g = StateGraph(dict)

    # ── Registrazione nodi ────────────────────────────────────────────────────
    g.add_node("preferences",           _preferences_node)
    g.add_node("constraint_classifier", _constraint_classifier_node)   # [NUOVO] Fase 1a
    g.add_node("dynamic_extension",     _dynamic_extension_node)       # [NUOVO] Fase 1b
    g.add_node("drafting",              _drafting_node)
    g.add_node("verification",          _verification_node)
    g.add_node("refinement",            _refinement_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    g.set_entry_point("preferences")

    # ── Edges fissi ───────────────────────────────────────────────────────────
    g.add_edge("preferences", "constraint_classifier")   # sempre
    g.add_edge("drafting",    "verification")            # sempre

    # ── Routing condizionale Fase 1a/1b ──────────────────────────────────────
    g.add_conditional_edges(
        "constraint_classifier",
        _route_after_classifier,
        {
            "drafting":          "drafting",
            "dynamic_extension": "dynamic_extension",
            "end":               END,
        },
    )

    g.add_conditional_edges(
        "dynamic_extension",
        _route_after_dynamic_extension,
        {
            "drafting": "drafting",
            "end":      END,
        },
    )

    # ── Routing Stages 3-4 ────────────────────────────────────────────────────
    g.add_conditional_edges(
        "verification",
        _route_after_verification,
        {"refinement": "refinement", "end": END},
    )

    g.add_conditional_edges(
        "refinement",
        _route_after_refinement,
        {"verify": "verification", "end": END},
    )

    return g.compile()


def run_pipeline(use_case: str = "A", input_file: str = None, rules_file: str = None) -> SmartSchedulerState:
    """Execute the full SmartScheduler pipeline and return the final state."""
    pipeline = build_pipeline()
    initial_state = SmartSchedulerState(use_case=use_case, input_file=input_file, rules_file=rules_file).model_dump()
    final_state_dict = pipeline.invoke(initial_state)
    return SmartSchedulerState(**final_state_dict)
