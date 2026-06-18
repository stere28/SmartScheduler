"""
SmartScheduler – LangGraph Pipeline
======================================
Wires together the four agents into a stateful graph with a refinement loop.
"""

from __future__ import annotations
from typing import Literal

from langgraph.graph import StateGraph, END

from models import SmartSchedulerState
from agents import (
    preferences_agent,
    drafting_agent,
    verification_agent,
    refinement_agent,
)
from config import MAX_REFINEMENT_ITERATIONS


# ── Node wrappers ──────────────────────────────────────────────────────────────
# LangGraph nodes receive and return dict-compatible state objects.

def _preferences_node(state: dict) -> dict:
    s = SmartSchedulerState(**state)
    result = preferences_agent(s)
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

def _route_after_verification(state: dict) -> Literal["refinement", "end"]:
    """After verification: if passed and not converged, refine; else end."""
    s = SmartSchedulerState(**state)
    if s.verification is None:
        return "end"
    if not s.verification.passed:
        # Hard constraint violation → re-draft (here we end; re-draft loop optional)
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

    # Register nodes
    g.add_node("preferences", _preferences_node)
    g.add_node("drafting",    _drafting_node)
    g.add_node("verification", _verification_node)
    g.add_node("refinement",  _refinement_node)

    # Entry point
    g.set_entry_point("preferences")

    # Edges
    g.add_edge("preferences", "drafting")
    g.add_edge("drafting",    "verification")

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


def run_pipeline(use_case: str = "A") -> SmartSchedulerState:
    """Execute the full SmartScheduler pipeline and return the final state."""
    pipeline = build_pipeline()
    initial_state = SmartSchedulerState(use_case=use_case).model_dump()
    final_state_dict = pipeline.invoke(initial_state)
    return SmartSchedulerState(**final_state_dict)
