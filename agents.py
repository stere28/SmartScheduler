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
from solver import solve_schedule, verify_hard_constraints

# ── Retry helper ──────────────────────────────────────────────────────────────

def _invoke_llm_with_retry(messages: list, max_retries: int = 3, delay: float = 2.0) -> str:
    """
    Invoke the LLM and return the raw text content.
    Retries up to `max_retries` times if the response is empty or not valid JSON.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            llm = get_llm()
            response = llm.invoke(messages)
            raw = response.content.strip()

            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            if not raw:
                raise ValueError("LLM returned an empty response.")

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


_VALID_SHIFTS = {"morning", "afternoon", "night"}
_MAX_UNAVAILABLE_DAYS = 2   # cap to keep problem feasible with 10 workers

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 – Preferences Agent
# ══════════════════════════════════════════════════════════════════════════════

#TODO correggere il prompt in modo tale da implementare le tecniche di prompt engineering 
_PREFERENCES_SYSTEM_PROMPT = """\
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

[ESEMPIO: PREFERENZE VARIE]
Testo di input: "Il Dr. Rossi (id: 123) è un chirurgo. Odia i turni di mattina ma adora le notti. Non può assolutamente lavorare nei fine settimana. Preferisce riposare il mercoledì. Non lavorerà nei giorni festivi in nessun caso, ma è disposto a fare 2 turni di emergenza."

Output:
{
  "workers": [
    {
      "worker_id": "123",
      "worker_name": "Dr. Rossi",
      "worker_type": "specialized",
      "preferred_shifts": ["night"],
      "avoided_shifts": ["morning"],
      "night_tolerance": true,
      "holiday_tolerance": false,
      "unavailable_days_of_week": [5, 6],
      "preferred_rest_day": 2,
      "emergency_coverage": 2
    }
  ]
}

[ESEMPIO: NESSUN VINCOLO]
Testo di input: "L'infermiera Giulia Bianchi (id: 456) del reparto di Pediatria è sempre disponibile. Non ha nessuna preferenza sui turni e lavora nei festivi senza problemi. Non è disponibile per turni extra di reperibilità."

Output:
{
  "workers": [
    {
      "worker_id": "456",
      "worker_name": "Giulia Bianchi",
      "worker_type": "standard",
      "preferred_shifts": [],
      "avoided_shifts": [],
      "night_tolerance": true,
      "holiday_tolerance": true,
      "unavailable_days_of_week": [],
      "preferred_rest_day": null,
      "emergency_coverage": 0
    }
  ]
}


For each worker described in the text, you must strictly follow this workflow:
1. IDENTIFICATION: Extract "worker_id", "worker_name", and "worker_type".
2. SHIFT ANALYSIS: Look for keywords indicating preferences or aversions for shifts (morning, afternoon, night). Assign them to "preferred_shifts" or "avoided_shifts", respectively.
3. TOLERANCE VERIFICATION: Check if the worker expresses a CATEGORICAL REFUSAL to work night shifts or holidays. If so, set the respective tolerance ("night_tolerance" or "holiday_tolerance") to false. Otherwise, keep the default as true.
4. DAY MAPPING: Distinguish between the absolute inability to work on a certain day (hard constraint -> "unavailable_days_of_week") and a simple preference for resting (soft constraint -> "preferred_rest_day"). Convert the days into their respective integers (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun).
5. EMERGENCY VERIFICATION: Identify how many extra emergency/on-call shifts per month the worker is willing to cover ("emergency_coverage").
6. SYNTHESIS AND OUTPUT: Write a brief summary of your reasoning and generate the final JSON.

Only output valid JSON, no extra text.
"""

def preferences_agent(state: SmartSchedulerState) -> SmartSchedulerState:
    """
    Stage 1: Parse worker preference descriptions and build WorkforcePreferences.
    """

    print("\n[Stage 1] Preferences Agent running…")

    use_case = state.use_case

    # Da file di testo, leggere le descrizioni dei lavoratori per il caso d'uso specifico.
    if use_case == "A":
        with open("Demo_A.txt", "r", encoding="utf-8") as f:
            raw_descriptions = f.read()
    else:
        with open("Demo_B.txt", "r", encoding="utf-8") as f:
            raw_descriptions = f.read()

    messages = [
        SystemMessage(content=_PREFERENCES_SYSTEM_PROMPT),
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
# STAGE 2a – LLM Drafting Agent
# ══════════════════════════════════════════════════════════════════════════════

#TODO correggere il prompt in modo tale da implementare le tecniche di prompt engineering 
_LLM_DRAFTING_SYSTEM_PROMPT = """\
You are a hospital shift scheduling expert. Your task is to generate a COMPLETE
monthly schedule for ALL workers for ALL 31 days.

The scheduling horizon has 31 days (day_index 0 to 30).
Shifts: 0=morning (08-14, 6h), 1=afternoon (14-20, 6h), 2=night (20-08, 12h).

HARD CONSTRAINTS (must ALL be satisfied):
- H1: Each shift (day × shift_type) must have at least 2 workers assigned.
- H2: Each worker can work AT MOST 1 shift per day.
- H3: After a night shift (shift_index=2), the worker cannot work the FOLLOWING morning (shift_index=0 next day).
- H4: After a night shift, the worker must have 2 FREE days (no assignment on day+1 and day+2).
- H5: Each worker's total workload for the month must equal exactly 25 units
       (morning=1 unit, afternoon=1 unit, night=2 units).
- H6: Each worker cannot exceed 36 hours per week.
       (morning=6h, afternoon=6h, night=12h; weeks are 7-day windows starting at day 0).
- H7: Workers have unavailable days of week (0=Mon,...,6=Sun).
       Do NOT assign any shift to a worker on their unavailable days.

IMPORTANT NOTES:
- A night shift counts as 2 workload units.
- With 31 days and 3 shifts × 2+ workers per shift, there are many assignments needed.
- Balance shifts evenly across days and workers.

Return ONLY a valid JSON object in this EXACT format (no extra text, no markdown):
{
  "assignments": [
    {"worker_id": "W01", "day_index": 0, "shift_index": 0},
    ...
  ]
}

Generate assignments for ALL workers across ALL 31 days respecting ALL constraints.
"""

DRAFTING_CORRECTION_SYSTEM_PROMPT = """\
You are a hospital scheduling expert.
A previous schedule attempt violated some HARD constraints.
You will receive a detailed violation report.

Your task is to acknowledge the violations and confirm you understand what
must be fixed, then the system will re-run the solver with your acknowledgement.

Output a JSON object with:
{
  "strategy_notes": "<explain what was wrong and how you would fix it>",
  "use_case": "A" or "B",
  "ready_to_solve": true
}

Only output valid JSON, no extra text.
"""


def llm_drafting_agent(state: SmartSchedulerState) -> SmartSchedulerState:
    """
    Stage 2a: Use the LLM to generate an initial schedule directly.
    The LLM produces a list of (worker_id, day_index, shift_index) assignments.
    """
    iteration_draft = state.iteration_draft
    print(f"\n[Stage 2a] LLM Drafting Agent – draft iteration {iteration_draft + 1}…")

    workers = state.preferences.workers
    summary = _summarise_preferences(workers)

    # Build feedback context from previous failed verification
    feedback_verification = None
    feedback_refinement = None
    if state.feedback_verification:
        feedback_verification = f"\nPrevious verification FAILED. Fix these violations:\n{state.feedback_verification}"
    if state.feedback_refinement:
        feedback_refinement = f"\nPrevious refinement feedback:\n{state.feedback_refinement}"

    messages = [
        SystemMessage(content=_LLM_DRAFTING_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps({
            "use_case": state.use_case,
            "num_workers": len(workers),
            "workers": summary,
            "feedback": feedback_verification or feedback_refinement or "No previous feedback – first attempt.",
        })),
    ]

    try:
        raw_json = _invoke_llm_with_retry(messages, max_retries=3)
        data = json.loads(raw_json)
        assignments_raw = data.get("assignments", [])

        assignments = [
            Assignment(
                worker_id=str(a["worker_id"]),
                day_index=int(a["day_index"]),
                shift_index=int(a["shift_index"]),
            )
            for a in assignments_raw
            if isinstance(a, dict)
               and "worker_id" in a and "day_index" in a and "shift_index" in a
               and 0 <= int(a["day_index"]) <= 30
               and 0 <= int(a["shift_index"]) <= 2
        ]

        if not assignments:
            raise ValueError("LLM returned an empty or invalid assignments list.")

        schedule = Schedule(assignments=assignments)
        print(f"    ✓ LLM generated schedule with {len(assignments)} assignments.")
        return state.model_copy(update={
            "schedule": schedule,
            "iteration_draft": iteration_draft + 1,
            "history": state.history + [
                f"[S2a] LLM draft #{iteration_draft + 1}: {len(assignments)} assignments."
            ],
        })

    except Exception as exc:
        print(f"    ✗ LLM drafting failed: {exc}")
        return state.model_copy(update={
            "schedule": None,
            "iteration_draft": iteration_draft + 1,
            "history": state.history + [
                f"[S2a] LLM draft #{iteration_draft + 1} FAILED: {exc}."
            ],
        })


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2b – OR-Tools Solver Drafting Agent
# ══════════════════════════════════════════════════════════════════════════════

def solver_drafting_agent(state: SmartSchedulerState) -> SmartSchedulerState:
    """
    Stage 2b: Use the OR-Tools CP-SAT solver to generate a provably feasible schedule.
    """
    print(f"\n[Stage 2b] OR-Tools Solver Drafting Agent")

    workers = state.preferences.workers

    schedule, sat_scores = solve_schedule(
        workers=workers,
        use_case=state.use_case,
        time_limit_seconds=120,
    )

    if schedule is None:
        print("   ✗ OR-Tools Solver returned INFEASIBLE.")
        return state.model_copy(update={
            "schedule": None,
            "history": state.history + [
                f"[S2b] Solver draft INFEASIBLE – no schedule generated."
            ],
        })

    min_sat = min(sat_scores.values()) if sat_scores else 0.0
    print(f"    ✓ Solver generated schedule ({len(schedule.assignments)} assignments). "
          f"Min sat: {min_sat:.1f}")
    return state.model_copy(update={
        "schedule": schedule,
        "constraint_feedback": None,   # clear previous feedback
        "history": state.history + [
            f"[S2b] Solver draft: {len(schedule.assignments)} assignments. "
            f"Min sat: {min_sat:.1f}."
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

# Human-readable descriptions of every hard constraint rule.
# Used to build the feedback message sent back to Stage 2.
_CONSTRAINT_DESCRIPTIONS: dict[str, str] = {
    "H1_staffing": (
        "H1 – Minimum staffing per shift: "
        "Use-case A requires ≥ 2 workers per shift; "
        "Use-case B requires ≥ 1 specialized + ≥ 1 standard + ≥ 3 total."
    ),
    "H2_one_shift_per_day": (
        "H2 – At most 1 shift per worker per day: "
        "a worker cannot be assigned to two different shifts on the same calendar day."
    ),
    "H3_no_consecutive": (
        "H3 – No afternoon → morning back-to-back: "
        "if a worker does the afternoon shift on day D, "
        "they cannot do the morning shift on day D+1."
    ),
    "H4_rest_after_night": (
        f"H4 – Mandatory rest after night shift: "
        f"after a night shift (shift index 2), the worker must have "
        f"{__import__('config').FREE_DAYS_AFTER_NIGHT} consecutive free days "
        f"(no assignment of any shift type)."
    ),
    "H5_workload": (
        f"H5 – Monthly workload target: "
        f"each worker must accumulate exactly "
        f"{__import__('config').TARGET_SHIFTS_MONTH} workload units "
        f"(morning/afternoon = 1 unit each, night = 2 units)."
    ),
    "H6_weekly_hours": (
        f"H6 – Weekly hours cap: "
        f"no worker may work more than "
        f"{__import__('config').MAX_HOURS_PER_WEEK} hours in any single calendar week "
        f"(morning = 6 h, afternoon = 6 h, night = 12 h)."
    ),
    "H7_weekly_rest": (
        "H7 – At least 1 rest day per week: "
        "every worker must have at least one day off in each calendar week."
    ),
    "H8_unavailability": (
        "H8 – Worker unavailability: "
        "workers with declared unavailable_days_of_week must never be assigned "
        "on those days of the week."
    ),
}


def _build_constraint_feedback(violations_by_rule: dict[str, list[str]]) -> str:
    """
    Build a human-readable violation report that Stage 2's LLM can understand
    and act upon.  The report lists every violated constraint with its
    definition and all specific instances.
    """
    lines = [
        "=" * 70,
        "HARD CONSTRAINT VIOLATION REPORT",
        "The schedule produced in Stage 2 violates the following rules.",
        "Each rule MUST be satisfied in the corrected schedule.",
        "=" * 70,
    ]
    for rule_key, instances in violations_by_rule.items():
        rule_desc = _CONSTRAINT_DESCRIPTIONS.get(rule_key, rule_key)
        lines.append(f"\n[{rule_key}] {rule_desc}")
        lines.append(f"  Violations ({len(instances)} instance(s)):")
        # cap to 20 instances to keep the prompt manageable
        for inst in instances[:20]:
            lines.append(inst)
        if len(instances) > 20:
            lines.append(f"  … and {len(instances) - 20} more.")
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def verification_agent(state: SmartSchedulerState) -> SmartSchedulerState:
    """
    Stage 3: Verify the schedule produced by Stage 2 against all hard
    constraints (via solver.verify_hard_constraints), compute per-worker
    fairness scores, and prepare feedback for Stage 2 if needed.

    Behaviour
    ---------
    • PASS  → computes fairness, stores VerificationReport, clears
              constraint_feedback.  The pipeline proceeds to Stage 4.
    • FAIL  → builds a structured violation report, stores it in
              constraint_feedback, and returns a failed VerificationReport.
              The pipeline routes back to Stage 2 (up to MAX_DRAFTING_ATTEMPTS).
    """
    print("\n[Stage 3] Verification Agent running…")

    # ── Guard: no schedule present ────────────────────────────────────────────
    if state.schedule is None:
        print("    ✗ No schedule to verify.")
        feedback = (
            "No schedule was produced by Stage 2. "
            "Please generate a complete assignment for all workers and all days."
        )
        return state.model_copy(update={
            "constraint_feedback": feedback,
            "verification": VerificationReport(
                passed=False,
                violations=[ConstraintViolation(description="No schedule available.")],
                fairness_scores={},
            ),
            "history": state.history + ["[S3] No schedule – feedback sent to S2."],
        })

    workers  = state.preferences.workers
    schedule = state.schedule
    use_case = state.use_case

    # ── Step 1: verify hard constraints via solver.py ─────────────────────────
    print("    Checking hard constraints via solver.verify_hard_constraints…")
    violations_by_rule = verify_hard_constraints(schedule, workers, use_case)

    total_violations = sum(len(v) for v in violations_by_rule.values())
    passed = total_violations == 0

    # Convert to flat ConstraintViolation list for the VerificationReport
    flat_violations: list[ConstraintViolation] = [
        ConstraintViolation(description=desc)
        for descs in violations_by_rule.values()
        for desc in descs
    ]

    # ── Step 2: handle FAIL → build feedback for Stage 2 ─────────────────────
    if not passed:
        n_rules = len(violations_by_rule)
        print(
            f"    ✗ {total_violations} hard constraint violation(s) across "
            f"{n_rules} rule(s):"
        )
        for rule_key, descs in violations_by_rule.items():
            print(f"      [{rule_key}] {len(descs)} violation(s)")
            for d in descs[:3]:
                print(f"        {d}")
            if len(descs) > 3:
                print(f"        … and {len(descs) - 3} more.")

        feedback = _build_constraint_feedback(violations_by_rule)

        return state.model_copy(update={
            "constraint_feedback": feedback,
            "verification": VerificationReport(
                passed=False,
                violations=flat_violations,
                fairness_scores={},
                min_satisfaction=0.0,
                most_disadvantaged_worker=None,
            ),
            "history": state.history + [
                f"[S3] FAILED – {total_violations} violation(s) in "
                f"{n_rules} rule(s). Feedback prepared for S2."
            ],
        })

    # ── Step 3: PASS → compute fairness and forward to Stage 4 ───────────────
    print("    ✓ All hard constraints satisfied.")
    fairness_scores = _compute_fairness(workers, schedule)
    min_sat   = min(fairness_scores.values()) if fairness_scores else 0.0
    max_sat   = max(fairness_scores.values()) if fairness_scores else 0.0
    
    sorted_workers = sorted(fairness_scores.items(), key=lambda x: x[1])
    worst_wid = sorted_workers[0][0] if sorted_workers else None
    
    second_worst_worker = None
    second_worst_satisfaction = 0.0
    if len(sorted_workers) > 1:
        second_worst_worker = sorted_workers[1][0]
        second_worst_satisfaction = sorted_workers[1][1]

    print(
        f"    Fairness – min: {min_sat:.1f}  max: {max_sat:.1f}  "
        f"delta: {max_sat - min_sat:.1f}  "
        f"most disadvantaged: {worst_wid}"
    )

    report = VerificationReport(
        passed=True,
        violations=[],
        fairness_scores=fairness_scores,
        min_satisfaction=min_sat,
        most_disadvantaged_worker=worst_wid,
        second_worst_worker=second_worst_worker,
        second_worst_satisfaction=second_worst_satisfaction,
    )

    return state.model_copy(update={
        "verification": report,
        "constraint_feedback": None,   # clear – no longer needed
        "history": state.history + [
            f"[S3] PASSED. Min sat: {min_sat:.1f}. "
            f"Max sat: {max_sat:.1f}. "
            f"Worst worker: {worst_wid}."
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
        min_raw = 0.0
        max_raw = 0.0

        worker_assignments = [a for a in schedule.assignments if a.worker_id == wp.worker_id]
        
        for a in worker_assignments:
            is_hol = _is_holiday(a.day_index)
            dow = DATES[a.day_index].weekday()
            
            # Actual score for this assignment
            shift_name = SHIFT_NAMES[a.shift_index]
            if shift_name in wp.preferred_shifts:
                raw += WEIGHT_PREFERRED_SHIFT
            if shift_name in wp.avoided_shifts:
                raw += WEIGHT_AVOIDED_SHIFT
            if a.shift_index == 2:
                raw += WEIGHT_NIGHT_TOLERANCE if wp.night_tolerance else WEIGHT_NIGHT_NO_TOLERANCE
            if is_hol:
                raw += WEIGHT_HOLIDAY_TOLERANCE if wp.holiday_tolerance else WEIGHT_HOLIDAY_NO_TOLERANCE
            if wp.preferred_rest_day is not None and dow == wp.preferred_rest_day:
                raw -= 5
                
            # Compute min and max possible for this specific day across all 3 shifts
            day_min = float('inf')
            day_max = float('-inf')
            for s in range(3):
                s_name = SHIFT_NAMES[s]
                s_score = 0.0
                if s_name in wp.preferred_shifts:
                    s_score += WEIGHT_PREFERRED_SHIFT
                if s_name in wp.avoided_shifts:
                    s_score += WEIGHT_AVOIDED_SHIFT
                if s == 2:
                    s_score += WEIGHT_NIGHT_TOLERANCE if wp.night_tolerance else WEIGHT_NIGHT_NO_TOLERANCE
                if is_hol:
                    s_score += WEIGHT_HOLIDAY_TOLERANCE if wp.holiday_tolerance else WEIGHT_HOLIDAY_NO_TOLERANCE
                if wp.preferred_rest_day is not None and dow == wp.preferred_rest_day:
                    s_score -= 5
                
                day_min = min(day_min, s_score)
                day_max = max(day_max, s_score)
                
            min_raw += day_min
            max_raw += day_max

        # Check if preferred rest day was actually granted at least once
        if wp.preferred_rest_day is not None:
            # How many preferred rest days exist in the month?
            total_rest_days_in_month = sum(1 for d in range(len(DATES)) if DATES[d].weekday() == wp.preferred_rest_day)
            
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
            
            # The max possible is that at least one is free (gets the bonus)
            if total_rest_days_in_month > 0:
                max_raw += WEIGHT_REST_DAY_MET

        if max_raw == min_raw:
            scores[wp.worker_id] = 50.0
        else:
            normalised = (raw - min_raw) / (max_raw - min_raw) * 100.0
            scores[wp.worker_id] = round(max(0.0, min(100.0, normalised)), 2)

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 – Refinement Agent
# ══════════════════════════════════════════════════════════════════════════════

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
    second_worst_id = verification.second_worst_worker
    second_worst_sat = verification.second_worst_satisfaction

    # -- GESTIONE EQUITÀ PERFETTA --
    fairness_scores = state.verification.fairness_scores
    if fairness_scores:
        max_sat = max(fairness_scores.values())
        min_sat = min(fairness_scores.values())
        if (max_sat - min_sat) <= 2.0:  # Tolleranza minima
            print("    ✓ All workers are equally satisfied (Delta ≤ 2). Converging early.")
            return state.model_copy(update={"converged": True})
        
    # LLM refinement strategy
    messages = [
        SystemMessage(content=(
            "You are an expert hospital scheduler. Analyze the worst-off worker."
            "Return a JSON object matching this schema: "
            "{'reasoning': 'string', 'shifts_to_avoid': [int], 'shifts_to_prefer': [int], 'weight_boost': int}."
            "Shifts: 0=morning, 1=afternoon, 2=night. weight_boost is 1-10."
        )),
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
    # Pulisce i blocchi markdown che l'LLM potrebbe aggiungere per errore
    clean_raw = raw.strip()
    if clean_raw.startswith("```json"):
        clean_raw = clean_raw[7:]
    if clean_raw.startswith("```"):
        clean_raw = clean_raw[3:]
    if clean_raw.endswith("```"):
        clean_raw = clean_raw[:-3]
        
    strategy = json.loads(clean_raw.strip())
    print(f"    Strategy for {worst_id}: {strategy.get('reasoning', 'N/A')}")

    # Re-solve with a floor to protect currently satisfied workers
    # Floor = second_worst_sat (as requested, don't worsen the second worst)
    floor = second_worst_sat

    schedule, sat_scores = solve_schedule(
        workers=workers,
        use_case=state.use_case,
        min_satisfaction_floor=floor,
        pinned_worst_worker_id=worst_id,
        pinned_min_floor=worst_sat + 0.1,
        strategy_hints=strategy,
        time_limit_seconds=120,
    )

    if schedule is None:
        print("    ✗ Refinement solver INFEASIBLE – converging.")
        return state.model_copy(update={"converged": True})

    new_min_sat = min(sat_scores.values()) if sat_scores else 0.0
    new_worst_sat = sat_scores.get(worst_id, 0.0)
    improved    = new_worst_sat > worst_sat

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
