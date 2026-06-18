"""
SmartScheduler – Agent Definitions
=====================================
Four LangChain / LangGraph agents implementing the four pipeline stages.
"""

from __future__ import annotations
import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llm_provider import get_llm
from config import SHIFT_NAMES, DATES
from models import (
    ShiftPreference, WorkforcePreferences,
    Schedule, Assignment,
    ConstraintViolation, VerificationReport,
    SmartSchedulerState,
)
from solver import solve_schedule

# ── Retry helper ──────────────────────────────────────────────────────────────

def _invoke_llm_with_retry(messages: list, max_retries: int = 3, delay: float = 2.0) -> str:
    """
    Invoke the LLM and return the raw text content.
    Retries up to `max_retries` times if the response is empty or not valid JSON.
    This is especially useful for smaller local models (e.g. llama3.2)
    that occasionally produce malformed or empty outputs.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            llm = get_llm()
            response = llm.invoke(messages)
            raw = response.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            if not raw:
                raise ValueError("LLM returned an empty response.")

            # Validate JSON
            json.loads(raw)
            return raw

        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            print(f"  [LLM] Attempt {attempt}/{max_retries} failed: {exc}. Retrying…")
            time.sleep(delay)

    raise RuntimeError(
        f"LLM failed to produce valid JSON after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


# ── Worker data sanitiser ──────────────────────────────────────────────────────

_VALID_SHIFTS = {"morning", "afternoon", "night"}
_MAX_UNAVAILABLE_DAYS = 2   # cap to keep problem feasible with 10 workers


def _sanitise_workers(workers_raw: list[dict]) -> list[dict]:
    """
    Post-process raw LLM output before constructing ShiftPreference objects.

    Smaller models often hallucinate:
    - Excessive unavailable_days_of_week (makes the CP-SAT model infeasible)
    - Non-shift strings in avoided_shifts (e.g. "consecutive holidays")
    - None instead of [] for list fields

    This function applies conservative defaults so the solver always has a
    valid, solvable input regardless of LLM output quality.
    """
    for w in workers_raw:
        # Coerce None → []
        for field in ("preferred_shifts", "avoided_shifts", "unavailable_days_of_week"):
            if w.get(field) is None:
                w[field] = []

        # Remove invalid shift names from shift lists
        w["preferred_shifts"] = [s for s in w["preferred_shifts"] if s in _VALID_SHIFTS]
        w["avoided_shifts"]   = [s for s in w["avoided_shifts"]   if s in _VALID_SHIFTS]

        # Cap unavailability: more than 2 days/week makes 25-shift target very hard
        unavail = w.get("unavailable_days_of_week", [])
        if isinstance(unavail, list) and len(unavail) > _MAX_UNAVAILABLE_DAYS:
            print(
                f"  [Sanitise] Worker {w.get('worker_id')}: "
                f"unavailable_days_of_week truncated from {unavail} to {unavail[:_MAX_UNAVAILABLE_DAYS]}"
            )
            w["unavailable_days_of_week"] = unavail[:_MAX_UNAVAILABLE_DAYS]

    return workers_raw


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 – Preferences Agent
# ══════════════════════════════════════════════════════════════════════════════

PREFERENCES_SYSTEM_PROMPT = """\
You are a scheduling assistant for a hospital. Extract STRUCTURED preferences from
natural language worker descriptions.

RULES (follow exactly):
1. preferred_shifts: list of SHIFT NAMES the worker EXPLICITLY prefers.
   Only use values: "morning", "afternoon", "night". Empty list [] if none stated.
2. avoided_shifts: list of SHIFT NAMES the worker EXPLICITLY wants to avoid.
   Only use values: "morning", "afternoon", "night". Empty list [] if none stated.
   NEVER put non-shift strings (like "consecutive holidays") here.
3. night_tolerance: true by default. Set false ONLY if the worker EXPLICITLY says
   they do NOT want or cannot do night shifts.
4. holiday_tolerance: true by default. Set false ONLY if the worker EXPLICITLY says
   they refuse holiday work.
5. unavailable_days_of_week: list of day integers (0=Mon,1=Tue,2=Wed,3=Thu,4=Fri,5=Sat,6=Sun)
   Set ONLY days when the worker says they are COMPLETELY UNAVAILABLE (e.g. "not available on Sundays").
   DO NOT add days for soft preferences like "prefers not to work on..." or "would rather avoid...".
   Use preferred_rest_day for soft day preferences instead.
   IMPORTANT: Keep this list to AT MOST 2 days. If in doubt, use empty list [].
6. preferred_rest_day: integer (0-6) or null. Use for SOFT preference ("prefers Saturdays off").
7. emergency_coverage: integer, extra shifts accepted per month. Default 0.

Return a JSON object with key "workers" containing the list.
Each object must have ALL fields: worker_id, worker_name, worker_type, preferred_shifts,
avoided_shifts, night_tolerance, holiday_tolerance, unavailable_days_of_week,
preferred_rest_day, emergency_coverage.

Only output valid JSON, no extra text.
"""


def preferences_agent(state: SmartSchedulerState) -> SmartSchedulerState:
    """
    Stage 1: Parse worker preference descriptions and build WorkforcePreferences.
    When called without real worker input (batch mode), uses the demo scenario.
    """
    print("\n[Stage 1] Preferences Agent running…")

    use_case = state.use_case

    if use_case == "A":
        raw_descriptions = _demo_preferences_A()
    else:
        raw_descriptions = _demo_preferences_B()

    messages = [
        SystemMessage(content=PREFERENCES_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps({"worker_descriptions": raw_descriptions})),
    ]

    raw_json = _invoke_llm_with_retry(messages)
    data = json.loads(raw_json)
    workers_raw = data["workers"]
    workers_raw = _sanitise_workers(workers_raw)
    workers = [ShiftPreference(**w) for w in workers_raw]
    preferences = WorkforcePreferences(workers=workers)

    state = state.model_copy(update={
        "preferences": preferences,
        "history": state.history + [f"[S1] Extracted preferences for {len(workers)} workers."],
    })
    print(f"    ✓ Preferences extracted for {len(workers)} workers.")
    return state



def _demo_preferences_A() -> list[dict]:
    """Synthetic preference descriptions for Use Case A (10 homogeneous workers)."""
    return [
        {
            "worker_id": "W01", "worker_name": "Alice Rossi", "worker_type": "standard",
            "description": (
                "Alice prefers morning shifts and would like to avoid night shifts whenever possible. "
                "She is not available on Sundays. Her preferred rest day is Saturday."
            )
        },
        {
            "worker_id": "W02", "worker_name": "Bruno Conti", "worker_type": "standard",
            "description": (
                "Bruno can work during weekends but not on consecutive holidays. "
                "He tolerates night shifts and is available for emergency coverage twice a month."
            )
        },
        {
            "worker_id": "W03", "worker_name": "Carla Esposito", "worker_type": "standard",
            "description": (
                "Carla prefers afternoon shifts. She does not mind night shifts "
                "but would like Christmas day off if possible."
            )
        },
        {
            "worker_id": "W04", "worker_name": "Davide Ferrari", "worker_type": "standard",
            "description": (
                "Davide prefers morning shifts. He strongly dislikes night shifts "
                "and is unavailable on Mondays."
            )
        },
        {
            "worker_id": "W05", "worker_name": "Elena Gallo", "worker_type": "standard",
            "description": (
                "Elena has no strong shift preferences. She accepts all shift types "
                "and holiday work. Preferred rest day is Wednesday."
            )
        },
        {
            "worker_id": "W06", "worker_name": "Francesco Bianchi", "worker_type": "standard",
            "description": (
                "Francesco prefers afternoon and night shifts. He is available on holidays "
                "and is willing to cover up to three emergency shifts per month."
            )
        },
        {
            "worker_id": "W07", "worker_name": "Giulia Marini", "worker_type": "standard",
            "description": (
                "Giulia prefers morning shifts and would like to avoid working on weekends. "
                "She has low tolerance for night shifts."
            )
        },
        {
            "worker_id": "W08", "worker_name": "Hector Romano", "worker_type": "standard",
            "description": (
                "Hector accepts any shift type. He prefers to have Thursdays as rest days. "
                "He is comfortable with night and holiday shifts."
            )
        },
        {
            "worker_id": "W09", "worker_name": "Irene Costa", "worker_type": "standard",
            "description": (
                "Irene prefers afternoon shifts. She would like to avoid night shifts but "
                "can handle one or two per month if needed."
            )
        },
        {
            "worker_id": "W10", "worker_name": "Luca Vitale", "worker_type": "standard",
            "description": (
                "Luca has no strong preferences. He is available all days and tolerates "
                "all shift types. He sees himself as a flexible team player."
            )
        },
    ]


def _demo_preferences_B() -> list[dict]:
    """Synthetic preference descriptions for Use Case B (10 standard + 6 specialized)."""
    standard = _demo_preferences_A()
    specialized = [
        {
            "worker_id": "S01", "worker_name": "Marco Ricci", "worker_type": "specialized",
            "description": (
                "Marco is a specialized doctor. He prefers morning shifts and has low "
                "tolerance for consecutive night shifts. He is unavailable on Sundays."
            )
        },
        {
            "worker_id": "S02", "worker_name": "Nadia Fontana", "worker_type": "specialized",
            "description": (
                "Nadia is a specialist. She prefers afternoon shifts and is comfortable "
                "with holiday work. Preferred rest day is Friday."
            )
        },
        {
            "worker_id": "S03", "worker_name": "Omar Greco", "worker_type": "specialized",
            "description": (
                "Omar works as a specialist. He has no strong preferences and is "
                "available for emergency coverage up to three times a month."
            )
        },
        {
            "worker_id": "S04", "worker_name": "Paola Serra", "worker_type": "specialized",
            "description": (
                "Paola is a specialized nurse. She prefers morning and avoids night shifts. "
                "She is unavailable on Tuesdays."
            )
        },
        {
            "worker_id": "S05", "worker_name": "Quirino De Luca", "worker_type": "specialized",
            "description": (
                "Quirino is a specialist who prefers afternoon shifts. He tolerates night "
                "shifts with low frequency and accepts holiday work."
            )
        },
        {
            "worker_id": "S06", "worker_name": "Rosa Amato", "worker_type": "specialized",
            "description": (
                "Rosa is a specialized technician. She has a strong preference for morning "
                "shifts and does not want to work on weekends or holidays."
            )
        },
    ]
    return standard + specialized


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 – Drafting Agent
# ══════════════════════════════════════════════════════════════════════════════

DRAFTING_SYSTEM_PROMPT = """\
You are a hospital scheduling expert with deep knowledge of OR-Tools CP-SAT.
Your task is to decide on the scheduling strategy and then delegate the actual
constraint solving to the OR-Tools solver.

Given a summary of worker preferences and constraints, output a JSON object with:
{
  "strategy_notes": "<short reasoning about the approach>",
  "use_case": "A" or "B",
  "ready_to_solve": true
}

Do NOT generate OR-Tools code yourself. The system will call the solver directly.
Only output valid JSON, no extra text.
"""


def drafting_agent(state: SmartSchedulerState) -> SmartSchedulerState:
    """
    Stage 2: Generate the initial schedule using the CP-SAT solver.
    The LLM provides strategic reasoning; the solver produces the actual schedule.
    """
    print("\n[Stage 2] Drafting Agent running…")

    preferences = state.preferences
    workers     = preferences.workers

    # LLM strategy consultation
    summary = _summarise_preferences(workers)
    messages = [
        SystemMessage(content=DRAFTING_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps({
            "use_case": state.use_case,
            "preferences_summary": summary,
            "num_workers": len(workers),
        })),
    ]
    raw = _invoke_llm_with_retry(messages)
    strategy = json.loads(raw)
    print(f"    Strategy: {strategy.get('strategy_notes', 'N/A')}")

    # Call the CP-SAT solver
    schedule, sat_scores = solve_schedule(
        workers=workers,
        use_case=state.use_case,
        time_limit_seconds=120,
    )

    if schedule is None:
        print("    ✗ Solver returned INFEASIBLE.")
        return state.model_copy(update={
            "history": state.history + ["[S2] Solver INFEASIBLE – no schedule generated."],
        })

    print(f"    ✓ Schedule generated ({len(schedule.assignments)} assignments).")
    return state.model_copy(update={
        "schedule": schedule,
        "history": state.history + [
            f"[S2] Schedule drafted. Min sat: {min(sat_scores.values()):.1f}",
        ],
    })


def _summarise_preferences(workers: list[ShiftPreference]) -> dict:
    summary: dict[str, Any] = {}
    for wp in workers:
        summary[wp.worker_id] = {
            "name": wp.worker_name,
            "type": wp.worker_type,
            "preferred": wp.preferred_shifts,
            "avoided":   wp.avoided_shifts,
            "night_ok":  wp.night_tolerance,
            "holiday_ok": wp.holiday_tolerance,
            "rest_day":  wp.preferred_rest_day,
        }
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 – Verification Agent
# ══════════════════════════════════════════════════════════════════════════════

def verification_agent(state: SmartSchedulerState) -> SmartSchedulerState:
    """
    Stage 3: Symbolically verify the schedule against hard constraints and
    compute per-worker fairness / satisfaction scores.
    """
    print("\n[Stage 3] Verification Agent running…")

    if state.schedule is None:
        print("    ✗ No schedule to verify.")
        return state.model_copy(update={
            "verification": VerificationReport(
                passed=False,
                violations=[ConstraintViolation(description="No schedule available.")],
                fairness_scores={},
            )
        })

    workers  = state.preferences.workers
    schedule = state.schedule
    use_case = state.use_case

    violations: list[ConstraintViolation] = []

    # Build lookup structures
    # assignment_map[worker_id][day_index] = shift_index (or None)
    from collections import defaultdict
    assign_map: dict[str, dict[int, int]] = defaultdict(dict)
    for a in schedule.assignments:
        assign_map[a.worker_id][a.day_index] = a.shift_index

    # Shift staffing map: {(day, shift) → list[worker_id]}
    shift_staff: dict[tuple, list] = defaultdict(list)
    for a in schedule.assignments:
        shift_staff[(a.day_index, a.shift_index)].append(a.worker_id)

    # -- Check H1: minimum staffing --
    for d in range(len(DATES)):
        for s in range(3):
            staff = shift_staff[(d, s)]
            if use_case == "A":
                if len(staff) < 2:
                    violations.append(ConstraintViolation(
                        description=f"Day {d} shift {SHIFT_NAMES[s]}: only {len(staff)} workers (need ≥2)."
                    ))
            else:
                n_std  = sum(1 for wid in staff if _worker_type(wid, workers) == "standard")
                n_spec = sum(1 for wid in staff if _worker_type(wid, workers) == "specialized")
                if n_spec < 1:
                    violations.append(ConstraintViolation(
                        description=f"Day {d} shift {SHIFT_NAMES[s]}: no specialized worker."
                    ))
                if n_std < 1:
                    violations.append(ConstraintViolation(
                        description=f"Day {d} shift {SHIFT_NAMES[s]}: no standard worker."
                    ))
                if len(staff) < 3:
                    violations.append(ConstraintViolation(
                        description=f"Day {d} shift {SHIFT_NAMES[s]}: only {len(staff)} total workers (need ≥3)."
                    ))

    # -- Check H2: at most 1 shift per day --
    for wp in workers:
        for d in range(len(DATES)):
            shifts_today = [a for a in schedule.assignments
                            if a.worker_id == wp.worker_id and a.day_index == d]
            if len(shifts_today) > 1:
                violations.append(ConstraintViolation(
                    description=f"Worker {wp.worker_id} assigned {len(shifts_today)} shifts on day {d}."
                ))

    # -- Check H4: 2 free days after night shift --
    from config import FREE_DAYS_AFTER_NIGHT
    for wp in workers:
        for d in range(len(DATES)):
            if assign_map[wp.worker_id].get(d) == 2:   # night shift
                for offset in range(1, FREE_DAYS_AFTER_NIGHT + 1):
                    if d + offset < len(DATES):
                        if d + offset in assign_map[wp.worker_id]:
                            violations.append(ConstraintViolation(
                                description=(
                                    f"Worker {wp.worker_id}: night on day {d}, "
                                    f"but works day {d+offset} (mandatory rest violated)."
                                )
                            ))

    # -- Check H5: total workload == 25 --
    from config import SHIFT_WEIGHT, TARGET_SHIFTS_MONTH
    for wp in workers:
        total = sum(
            SHIFT_WEIGHT[a.shift_index]
            for a in schedule.assignments if a.worker_id == wp.worker_id
        )
        if total != TARGET_SHIFTS_MONTH:
            violations.append(ConstraintViolation(
                description=(
                    f"Worker {wp.worker_id}: workload={total}, expected={TARGET_SHIFTS_MONTH}."
                )
            ))

    # -- Check H6: weekly hours ≤ 36 --
    import math
    from config import SHIFT_HOURS, MAX_HOURS_PER_WEEK
    num_weeks = math.ceil(len(DATES) / 7)
    for wp in workers:
        for wk in range(num_weeks):
            day_start = wk * 7
            day_end   = min(day_start + 7, len(DATES))
            hours = sum(
                SHIFT_HOURS[a.shift_index]
                for a in schedule.assignments
                if a.worker_id == wp.worker_id and day_start <= a.day_index < day_end
            )
            if hours > MAX_HOURS_PER_WEEK:
                violations.append(ConstraintViolation(
                    description=f"Worker {wp.worker_id} week {wk}: {hours}h > {MAX_HOURS_PER_WEEK}h."
                ))

    passed = len(violations) == 0

    # -- Compute fairness scores --
    fairness_scores = _compute_fairness(workers, schedule)
    min_sat  = min(fairness_scores.values()) if fairness_scores else 0.0
    worst_wid = min(fairness_scores, key=fairness_scores.get) if fairness_scores else None

    report = VerificationReport(
        passed=passed,
        violations=violations,
        fairness_scores=fairness_scores,
        min_satisfaction=min_sat,
        most_disadvantaged_worker=worst_wid,
    )

    if passed:
        print(f"    ✓ All hard constraints satisfied. Min satisfaction: {min_sat:.1f}")
    else:
        print(f"    ✗ {len(violations)} hard constraint violation(s) found.")
        for v in violations[:5]:
            print(f"      • {v.description}")

    return state.model_copy(update={
        "verification": report,
        "history": state.history + [
            f"[S3] Verification {'PASSED' if passed else 'FAILED'}. "
            f"Violations: {len(violations)}. Min sat: {min_sat:.1f}. "
            f"Worst: {worst_wid}."
        ],
    })


def _worker_type(worker_id: str, workers: list[ShiftPreference]) -> str:
    for w in workers:
        if w.worker_id == worker_id:
            return w.worker_type
    return "standard"


def _compute_fairness(
    workers: list[ShiftPreference], schedule: Schedule
) -> dict[str, float]:
    """Compute a 0-100 satisfaction score per worker based on the schedule."""
    from config import (
        WEIGHT_PREFERRED_SHIFT, WEIGHT_AVOIDED_SHIFT,
        WEIGHT_NIGHT_TOLERANCE, WEIGHT_NIGHT_NO_TOLERANCE,
        WEIGHT_HOLIDAY_TOLERANCE, WEIGHT_HOLIDAY_NO_TOLERANCE,
        WEIGHT_REST_DAY_MET,
    )
    from solver import _is_holiday

    scores: dict[str, float] = {}
    for wp in workers:
        raw = 0.0
        worker_assignments = [a for a in schedule.assignments if a.worker_id == wp.worker_id]
        for a in worker_assignments:
            shift_name = SHIFT_NAMES[a.shift_index]
            is_hol     = _is_holiday(a.day_index)
            dow        = DATES[a.day_index].weekday()

            if shift_name in wp.preferred_shifts:
                raw += WEIGHT_PREFERRED_SHIFT
            if shift_name in wp.avoided_shifts:
                raw += WEIGHT_AVOIDED_SHIFT

            if a.shift_index == 2:
                raw += WEIGHT_NIGHT_TOLERANCE if wp.night_tolerance else WEIGHT_NIGHT_NO_TOLERANCE

            if is_hol:
                raw += WEIGHT_HOLIDAY_TOLERANCE if wp.holiday_tolerance else WEIGHT_HOLIDAY_NO_TOLERANCE

            if wp.preferred_rest_day is not None and dow == wp.preferred_rest_day:
                raw -= 5   # penalise working on preferred rest day

        # Check if preferred rest day was actually granted at least once
        if wp.preferred_rest_day is not None:
            rest_days_free = sum(
                1 for d in range(len(DATES))
                if DATES[d].weekday() == wp.preferred_rest_day
                and not any(
                    a.worker_id == wp.worker_id and a.day_index == d
                    for a in schedule.assignments
                )
            )
            if rest_days_free > 0:
                raw += WEIGHT_REST_DAY_MET

        # Normalise to [0, 100]
        normalised = max(0.0, min(100.0, (raw + 200) / 4))
        scores[wp.worker_id] = round(normalised, 2)

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 – Refinement Agent
# ══════════════════════════════════════════════════════════════════════════════

REFINEMENT_SYSTEM_PROMPT = """\
You are a hospital scheduling optimizer. You are given the current schedule's
fairness report and must decide on a refinement strategy.

Focus on improving the satisfaction of the LEAST satisfied worker without 
worsening the satisfaction of other workers.

Output a JSON object:
{
  "focus_worker": "<worker_id>",
  "strategy": "<description of refinement approach>",
  "prioritize_shifts": ["morning"|"afternoon"|"night"],
  "reduce_shifts": ["morning"|"afternoon"|"night"]
}

Only output valid JSON, no extra text.
"""


def refinement_agent(state: SmartSchedulerState) -> SmartSchedulerState:
    """
    Stage 4: Iteratively refine the schedule to improve fairness.
    """
    print(f"\n[Stage 4] Refinement Agent – iteration {state.iteration + 1}…")

    verification = state.verification
    workers      = state.preferences.workers

    if not verification or not verification.most_disadvantaged_worker:
        print("    No disadvantaged worker identified – skipping refinement.")
        return state.model_copy(update={"converged": True})

    worst_id  = verification.most_disadvantaged_worker
    worst_sat = verification.min_satisfaction

    # LLM refinement strategy
    messages = [
        SystemMessage(content=REFINEMENT_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps({
            "focus_worker": worst_id,
            "current_min_satisfaction": worst_sat,
            "fairness_scores": verification.fairness_scores,
            "worker_preferences": {
                wp.worker_id: {
                    "preferred": wp.preferred_shifts,
                    "avoided":   wp.avoided_shifts,
                    "night_ok":  wp.night_tolerance,
                    "holiday_ok": wp.holiday_tolerance,
                }
                for wp in workers
            },
        })),
    ]
    raw = _invoke_llm_with_retry(messages)
    strategy = json.loads(raw)
    print(f"    Strategy for {worst_id}: {strategy.get('strategy', 'N/A')}")

    # Re-solve with a floor to protect currently satisfied workers
    # Floor = current min_sat - small tolerance (allow slight regression in others)
    floor = max(0.0, worst_sat - 5.0)

    schedule, sat_scores = solve_schedule(
        workers=workers,
        use_case=state.use_case,
        min_satisfaction_floor=floor,
        pinned_worst_worker_id=worst_id,
        time_limit_seconds=120,
    )

    if schedule is None:
        print("    ✗ Refinement solver INFEASIBLE – converging.")
        return state.model_copy(update={"converged": True})

    new_min_sat = min(sat_scores.values()) if sat_scores else 0.0
    improved    = new_min_sat > worst_sat

    print(f"    Min sat: {worst_sat:.1f} → {new_min_sat:.1f} ({'↑ improved' if improved else '↔ no change'})")

    if not improved:
        print("    No improvement achievable – converging.")
        return state.model_copy(update={"converged": True, "iteration": state.iteration + 1})

    return state.model_copy(update={
        "schedule":  schedule,
        "iteration": state.iteration + 1,
        "converged": False,
        "history":   state.history + [
            f"[S4] iter={state.iteration+1} min_sat: {worst_sat:.1f}→{new_min_sat:.1f}. "
            f"focus={worst_id}."
        ],
    })
