"""
SmartScheduler – LangGraph Pipeline
======================================
Wires together the agents into a stateful graph.
"""

from __future__ import annotations
from typing import Literal

from langgraph.graph import StateGraph, END

from models import SmartSchedulerState
from agents import (
    preferences_agent,
    llm_drafting_agent,
    solver_drafting_agent,
    verification_agent,
    refinement_agent,
)
from config import MAX_REFINEMENT_ITERATIONS, MAX_DRAFT_ITERATIONS


# ── Node wrappers ──────────────────────────────────────────────────────────────

def _preferences_node(state: dict) -> dict:
    s = SmartSchedulerState(**state)
    result = preferences_agent(s)
    return result.model_dump()


def _llm_drafting_node(state: dict) -> dict:
    s = SmartSchedulerState(**state)
    result = llm_drafting_agent(s)
    return result.model_dump()


def _solver_drafting_node(state: dict) -> dict:
    s = SmartSchedulerState(**state)
    result = solver_drafting_agent(s)
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

def _route_after_verification(state: dict,) -> Literal["refinement", "drafting_llm", "drafting_solver", "end"]:
    """
    After verification:
    """
    s = SmartSchedulerState(**state)

    if s.verification_passed:
        return "refinement"

    if s.iteration_draft < MAX_DRAFT_ITERATIONS:
        return "drafting_llm"
    return "drafting_solver"


def _route_after_refinement(state: dict) -> Literal["verification", "end"]:
    """
    After refinement: re-verify, unless converged.
    """
    s = SmartSchedulerState(**state)
    if s.converged:
        return "end"
    if s.iteration_draft < MAX_REFINEMENT_ITERATIONS:
        return "drafting_llm"
    return "drafting_solver"


# ── Build graph ────────────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    g = StateGraph(dict)

    # Register nodes
    g.add_node("preferences",       _preferences_node)
    g.add_node("drafting_llm",      _llm_drafting_node)       
    g.add_node("drafting_solver",   _solver_drafting_node)    
    g.add_node("verification",      _verification_node)       
    g.add_node("refinement",        _refinement_node)         

    # Entry point
    g.set_entry_point("preferences")

    g.add_edge("preferences", "drafting_llm")
    g.add_edge("drafting_llm",    "verification")
    g.add_edge("drafting_solver", END)
    # Verification → refinement | back to drafting | end
    g.add_conditional_edges(
        "verification",
        _route_after_verification,
        {
            "refinement":      "refinement",
            "drafting_llm":    "drafting_llm",
            "drafting_solver": "drafting_solver"
        },
    )

    g.add_conditional_edges(
        "refinement",
        _route_after_refinement,
        {"verification": "verification", 
         "end": END},
    )

    return g.compile()


def run_pipeline(use_case: str = "A") -> SmartSchedulerState:
    """Execute the full SmartScheduler pipeline and return the final state."""
    pipeline = build_pipeline()
    initial_state = SmartSchedulerState(use_case=use_case).model_dump()
    final_state_dict = pipeline.invoke(initial_state)
    return SmartSchedulerState(**final_state_dict)
