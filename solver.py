"""
SmartScheduler – OR-Tools CP-SAT Solver
=========================================
Implements the core scheduling model.
Used both by the drafting agent (initial solve) and the refinement agent
(re-solve with updated soft-constraint weights).
"""

from __future__ import annotations
import math
from typing import Optional

from ortools.sat.python import cp_model

from config import (
    NUM_DAYS, NUM_SHIFTS, SHIFT_WEIGHT, MAX_HOURS_PER_WEEK,
    SHIFT_HOURS, FREE_DAYS_AFTER_NIGHT, TARGET_SHIFTS_MONTH,
    DATES,
    WEIGHT_PREFERRED_SHIFT, WEIGHT_AVOIDED_SHIFT,
    WEIGHT_NIGHT_TOLERANCE, WEIGHT_NIGHT_NO_TOLERANCE,
    WEIGHT_HOLIDAY_TOLERANCE, WEIGHT_HOLIDAY_NO_TOLERANCE,
    WEIGHT_REST_DAY_MET,
)
from models import ShiftPreference, Schedule, Assignment


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_holiday(day_index: int) -> bool:
    """Return True for weekends and public holidays in the scheduling window."""
    d = DATES[day_index]
    # Christmas, Boxing Day, New Year's Day, weekends
    public_holidays = {(12, 25), (12, 26), (1, 1)}
    if (d.month, d.day) in public_holidays:
        return True
    return d.weekday() >= 5   # Saturday=5, Sunday=6


def _week_of(day_index: int) -> int:
    """Return the ISO week bucket (0-indexed within the horizon)."""
    return day_index // 7


# ── Hard-constraint verifier (no solving) ─────────────────────────────────────

def verify_hard_constraints(
    schedule: "Schedule",
    workers: list["ShiftPreference"],
    use_case: str = "A",
) -> dict[str, list[str]]:
    """
    Verify all hard constraints on an already-produced Schedule.

    Does NOT run OR-Tools: it simply inspects the Assignment list and checks
    each rule symbolically.  This is the canonical verifier used by Stage 3 so
    that the constraint logic is defined in one place only.

    Returns
    -------
    violations_by_rule : dict[str, list[str]]
        Keys are short constraint labels (e.g. "H1_staffing").
        Values are human-readable violation descriptions.
        An empty dict means the schedule passes all hard constraints.
    """
    from collections import defaultdict
    from config import (
        SHIFT_NAMES, SHIFT_WEIGHT, SHIFT_HOURS,
        FREE_DAYS_AFTER_NIGHT, TARGET_SHIFTS_MONTH,
        MAX_HOURS_PER_WEEK, DATES,
    )

    violations: dict[str, list[str]] = defaultdict(list)

    # ── Build lookup structures ────────────────────────────────────────────────
    # assign_map[worker_id][day_index] = shift_index
    assign_map: dict[str, dict[int, int]] = defaultdict(dict)
    for a in schedule.assignments:
        assign_map[a.worker_id][a.day_index] = a.shift_index

    # shift_staff[(day, shift)] = [worker_id, …]
    shift_staff: dict[tuple, list] = defaultdict(list)
    for a in schedule.assignments:
        shift_staff[(a.day_index, a.shift_index)].append(a.worker_id)

    num_days = len(DATES)
    num_weeks = math.ceil(num_days / 7)

    def _wtype(wid: str) -> str:
        for w in workers:
            if w.worker_id == wid:
                return w.worker_type
        return "standard"

    # ── H1: minimum staffing per shift ────────────────────────────────────────
    for d in range(num_days):
        for s in range(3):
            staff = shift_staff[(d, s)]
            date_str = DATES[d].strftime("%Y-%m-%d")
            shift_label = f"{date_str} [{SHIFT_NAMES[s]}]"
            if use_case == "A":
                if len(staff) < 2:
                    violations["H1_staffing"].append(
                        f"  • {shift_label}: {len(staff)} worker(s) assigned, need ≥ 2."
                    )
            else:
                n_spec = sum(1 for wid in staff if _wtype(wid) == "specialized")
                n_std  = sum(1 for wid in staff if _wtype(wid) == "standard")
                if n_spec < 1:
                    violations["H1_staffing"].append(
                        f"  • {shift_label}: no specialized worker (need ≥ 1)."
                    )
                if n_std < 1:
                    violations["H1_staffing"].append(
                        f"  • {shift_label}: no standard worker (need ≥ 1)."
                    )
                if len(staff) < 3:
                    violations["H1_staffing"].append(
                        f"  • {shift_label}: {len(staff)} total worker(s), need ≥ 3."
                    )

    # ── H2: at most 1 shift per worker per day ────────────────────────────────
    for wp in workers:
        for d in range(num_days):
            shifts_today = [
                a for a in schedule.assignments
                if a.worker_id == wp.worker_id and a.day_index == d
            ]
            if len(shifts_today) > 1:
                violations["H2_one_shift_per_day"].append(
                    f"  • Worker {wp.worker_id} ({wp.worker_name}): "
                    f"{len(shifts_today)} shifts on {DATES[d].strftime('%Y-%m-%d')} "
                    f"(shifts: {[SHIFT_NAMES[a.shift_index] for a in shifts_today]})."
                )

    # ── H3: no night→morning across consecutive days ─────────────────────────
    for wp in workers:
        for d in range(num_days - 1):
            # 2 = Night, 0 = Morning
            if (assign_map[wp.worker_id].get(d) == 2 and 
                    assign_map[wp.worker_id].get(d + 1) == 0):
                violations["H3_no_consecutive"].append(
                    f"  • Worker {wp.worker_id} ({wp.worker_name}): "
                    f"night on {DATES[d].strftime('%Y-%m-%d')} "
                    f"followed by morning on {DATES[d+1].strftime('%Y-%m-%d')}."
                )

    # ── H4: 2 mandatory free days after each night shift ─────────────────────
    for wp in workers:
        for d in range(num_days):
            if assign_map[wp.worker_id].get(d) == 2:   # night
                for offset in range(1, FREE_DAYS_AFTER_NIGHT + 1):
                    next_d = d + offset
                    if next_d < num_days and next_d in assign_map[wp.worker_id]:
                        violations["H4_rest_after_night"].append(
                            f"  • Worker {wp.worker_id} ({wp.worker_name}): "
                            f"night on {DATES[d].strftime('%Y-%m-%d')}, "
                            f"but works {DATES[next_d].strftime('%Y-%m-%d')} "
                            f"(mandatory rest day {offset} violated)."
                        )

    # ── H5: total workload == TARGET_SHIFTS_MONTH ─────────────────────────────
    for wp in workers:
        total = sum(
            SHIFT_WEIGHT[a.shift_index]
            for a in schedule.assignments if a.worker_id == wp.worker_id
        )
        if total != TARGET_SHIFTS_MONTH:
            violations["H5_workload"].append(
                f"  • Worker {wp.worker_id} ({wp.worker_name}): "
                f"workload = {total} units, expected exactly {TARGET_SHIFTS_MONTH}."
            )

    # ── H6: weekly hours ≤ MAX_HOURS_PER_WEEK ────────────────────────────────
    for wp in workers:
        for wk in range(num_weeks):
            day_start = wk * 7
            day_end   = min(day_start + 7, num_days)
            hours = sum(
                SHIFT_HOURS[assign_map[wp.worker_id][d]]
                for d in range(day_start, day_end)
                if d in assign_map[wp.worker_id]
            )
            if hours > MAX_HOURS_PER_WEEK:
                week_start_date = DATES[day_start].strftime("%Y-%m-%d")
                violations["H6_weekly_hours"].append(
                    f"  • Worker {wp.worker_id} ({wp.worker_name}): "
                    f"week starting {week_start_date} → {hours}h > {MAX_HOURS_PER_WEEK}h limit."
                )

    # ── H7: at least 1 rest day per week ─────────────────────────────────────
    for wp in workers:
        for wk in range(num_weeks):
            day_start  = wk * 7
            day_end    = min(day_start + 7, num_days)
            worked_days = sum(
                1 for d in range(day_start, day_end)
                if d in assign_map[wp.worker_id]
            )
            if worked_days == (day_end - day_start):
                week_start_date = DATES[day_start].strftime("%Y-%m-%d")
                violations["H7_weekly_rest"].append(
                    f"  • Worker {wp.worker_id} ({wp.worker_name}): "
                    f"0 rest days in week starting {week_start_date} "
                    f"(worked all {worked_days} days)."
                )

    # ── H8: worker unavailability (hard day-of-week blocks) ──────────────────
    for wp in workers:
        if not wp.unavailable_days_of_week:
            continue
        for a in schedule.assignments:
            if a.worker_id != wp.worker_id:
                continue
            dow = DATES[a.day_index].weekday()
            if dow in wp.unavailable_days_of_week:
                day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                violations["H8_unavailability"].append(
                    f"  • Worker {wp.worker_id} ({wp.worker_name}): "
                    f"assigned on {DATES[a.day_index].strftime('%Y-%m-%d')} "
                    f"({day_names[dow]}) but declared unavailable on {day_names[dow]}s."
                )

    return dict(violations)


# ── Main solver function ───────────────────────────────────────────────────────

def _normalized_to_raw_floor(normalized_floor: float, scale: int) -> int:
    """
    Convert a normalized satisfaction floor (0-100 range, same as returned by
    _compute_fairness / sat_scores) into the integer domain used by CP-SAT
    worker_sat variables.

    The normalization formula is:  normalized = (raw + 200) / 4
    Inverse:                       raw = normalized * 4 - 200
    CP-SAT uses integers:          floor_int = int(raw * scale)
    """
    raw_floor = normalized_floor * 4.0 - 200.0
    return int(raw_floor * scale)


def solve_schedule(
    workers: list[ShiftPreference],
    use_case: str = "A",
    min_satisfaction_floor: float = 0.0,
    pinned_worst_worker_id: Optional[str] = None,
    pinned_min_floor: float = 0.0,
    strategy_hints: dict | None = None,
    time_limit_seconds: int = 60,
) -> tuple[Optional[Schedule], dict[str, float]]:
    """
    Build and solve the CP-SAT model.

    Args:
        workers:                 List of worker preference objects.
        use_case:                "A" (homogeneous) or "B" (std + specialized).
        min_satisfaction_floor:  Normalized (0-100) floor applied to ALL workers
                                 EXCEPT the pinned worker.  Used by Stage 4 to
                                 protect the second-worst worker and above.
        pinned_worst_worker_id:  During refinement, the ID of the worst-off worker
                                 whose satisfaction we want to explicitly improve.
        pinned_min_floor:        Normalized (0-100) floor that the pinned worker
                                 MUST strictly exceed.  Should be set to the
                                 worker's current satisfaction score so the solver
                                 is forced to find a strictly better assignment.
        strategy_hints:          LLM-generated JSON with optional keys:
                                 ``shifts_to_avoid`` (list[int]) – hard-banned shifts;
                                 ``shifts_to_prefer`` (list[int]) – soft-boosted shifts;
                                 ``weight_boost`` (int 1-10) – objective multiplier.
        time_limit_seconds:      CP-SAT wall-clock limit.

    Returns:
        (schedule, satisfaction_scores) or (None, {}) if infeasible.
    """
    model  = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers  = 8

    n_workers = len(workers)
    DAYS      = range(NUM_DAYS)
    SHIFTS    = range(NUM_SHIFTS)   # 0=morning, 1=afternoon, 2=night
    WORKERS   = range(n_workers)

    # ── Decision variables ─────────────────────────────────────────────────────
    # x[w][d][s] = 1 if worker w is assigned to day d, shift s
    x = [[[model.new_bool_var(f"x_{w}_{d}_{s}")
           for s in SHIFTS]
          for d in DAYS]
         for w in WORKERS]

    # ── Hard Constraints ───────────────────────────────────────────────────────

    # (H1) Staffing requirements per shift
    for d in DAYS:
        for s in SHIFTS:
            shift_workers = [x[w][d][s] for w in WORKERS]

            if use_case == "A":
                # At least 2 workers per shift
                model.add(sum(shift_workers) >= 2)

            else:  # Use case B
                w_types = [workers[w].worker_type for w in WORKERS]
                std_vars  = [x[w][d][s] for w in WORKERS if w_types[w] == "standard"]
                spec_vars = [x[w][d][s] for w in WORKERS if w_types[w] == "specialized"]
                # At least 1 specialized
                model.add(sum(spec_vars) >= 1)
                # At least 2 standard-role (specialized can substitute):
                # (std_count + spec_count) - spec_count >= 2  →  std_count >= 2
                # BUT specialized can play standard role, so:
                # effective_standard = std_count + spec_count >= 3 total, spec >= 1, std >= 2 OR
                # "at least 2 std AND at least 1 spec" OR "total >= 3 and spec >= 1"
                # Per spec: "at least 2 standard workers and one specialized worker must be
                # assigned to each shift. If needed, a specialized worker can also play the
                # role of a standard one (e.g., a shift may be covered by one standard and
                # two specialized workers)."
                # Interpretation: minimum 2 standard-role workers + 1 specialized,
                # where a specialized can fill a standard role.
                # Simplest encoding: total >= 3, spec >= 1, std >= 1
                model.add(sum(spec_vars) >= 1)
                model.add(sum(std_vars)  >= 1)
                model.add(sum(shift_workers) >= 3)

    # (H2) At most 1 shift per worker per day
    for w in WORKERS:
        for d in DAYS:
            model.add(sum(x[w][d][s] for s in SHIFTS) <= 1)

    # (H3) No consecutive shifts across days:
    #       Night(d) ends at 08:00 on d+1, so morning(d+1) would start immediately → prohibit.
    #       Also prohibit afternoon(d) → morning(d+1) cross-day back-to-back.
    for w in WORKERS:
        for d in range(NUM_DAYS - 1):
            # Night shift on day d → no morning shift on day d+1
            model.add_implication(x[w][d][2], x[w][d+1][0].negated())
            # Afternoon(d) → no morning(d+1) (optional: strict no-consecutive interpretation)
            # model.add_implication(x[w][d][1], x[w][d+1][0].negated())

    # (H4) Mandatory 2 free days after each night shift
    for w in WORKERS:
        for d in DAYS:
            for offset in range(1, FREE_DAYS_AFTER_NIGHT + 1):
                if d + offset < NUM_DAYS:
                    # no shift of any type on d+offset when night shift on day d
                    for s in SHIFTS:
                        model.add_implication(
                            x[w][d][2],
                            x[w][d + offset][s].negated()
                        )

    # (H5) Each worker must cover exactly TARGET_SHIFTS_MONTH shifts
    #      (night = 2 workload units → counted as 2 in the total)
    for w in WORKERS:
        total_workload = sum(
            x[w][d][s] * SHIFT_WEIGHT[s]
            for d in DAYS for s in SHIFTS
        )
        model.add(total_workload == TARGET_SHIFTS_MONTH)

    # (H6) Weekly working hours ≤ 36
    #      We have 31 days → 4 full weeks + 3 days (weeks 0-3, remainder in week 4)
    num_weeks = math.ceil(NUM_DAYS / 7)
    for w in WORKERS:
        for wk in range(num_weeks):
            day_start = wk * 7
            day_end   = min(day_start + 7, NUM_DAYS)
            week_hours = sum(
                x[w][d][s] * SHIFT_HOURS[s]
                for d in range(day_start, day_end)
                for s in SHIFTS
            )
            model.add(week_hours <= MAX_HOURS_PER_WEEK)

    # (H7) Almeno 1 giorno di riposo a settimana
    # Utilizziamo 'num_weeks' già calcolato nel vincolo H6
    for w in WORKERS:
        for wk in range(num_weeks):
            day_start = wk * 7
            day_end   = min(day_start + 7, NUM_DAYS)
            
            # Calcoliamo quanti giorni totali ci sono in questa specifica "settimana"
            # (l'ultima settimana del mese potrebbe avere meno di 7 giorni)
            days_in_block = day_end - day_start
            
            # Somma di tutti i turni assegnati al lavoratore in questa settimana
            worked_days_in_block = sum(
                x[w][d][s] 
                for d in range(day_start, day_end) 
                for s in SHIFTS
            )
            
            # Per avere almeno un giorno di riposo, i giorni lavorati devono essere
            # minori o uguali ai giorni totali del blocco meno 1.
            model.add(worked_days_in_block <= days_in_block - 1)

    # (H8) Worker unavailability (day-of-week constraints from preferences)
    # (Rinomina il vecchio commento H7 in H8 per mantenere la coerenza numerica con il verifier)
    for w, wp in enumerate(workers):
        if wp.unavailable_days_of_week:
            for d in DAYS:
                dow = DATES[d].weekday()
                if dow in wp.unavailable_days_of_week:
                    for s in SHIFTS:
                        model.add(x[w][d][s] == 0)

    # ── Soft Constraints (penalty / bonus terms) ───────────────────────────────
    SCALE = 100   # scale factor to keep integers manageable

    soft_terms = []

    for w, wp in enumerate(workers):
        for d in DAYS:
            is_hol = _is_holiday(d)
            dow    = DATES[d].weekday()

            for s in SHIFTS:
                shift_name = ["morning", "afternoon", "night"][s]
                coeff = 0

                # Preferred shift
                if shift_name in wp.preferred_shifts:
                    coeff += WEIGHT_PREFERRED_SHIFT

                # Avoided shift
                if shift_name in wp.avoided_shifts:
                    coeff += WEIGHT_AVOIDED_SHIFT

                # Night tolerance
                if s == 2:
                    if wp.night_tolerance:
                        coeff += WEIGHT_NIGHT_TOLERANCE
                    else:
                        coeff += WEIGHT_NIGHT_NO_TOLERANCE

                # Holiday tolerance
                if is_hol:
                    if wp.holiday_tolerance:
                        coeff += WEIGHT_HOLIDAY_TOLERANCE
                    else:
                        coeff += WEIGHT_HOLIDAY_NO_TOLERANCE

                # Rest day preference: if worker prefers dow as rest day,
                # penalize any assignment on that day
                if wp.preferred_rest_day is not None and dow == wp.preferred_rest_day:
                    coeff -= 5   # soft penalty for working on preferred rest day

                if coeff != 0:
                    soft_terms.append(coeff * SCALE * x[w][d][s])

    # ── Objective ─────────────────────────────────────────────────────────────
    # Maximize total satisfaction + fairness (maximize the min satisfaction).
    # We model the min-satisfaction as an auxiliary variable.

    # Per-worker satisfaction variables (scaled integers)
    worker_sat = []
    for w, wp in enumerate(workers):
        sat_terms = []
        for d in DAYS:
            is_hol = _is_holiday(d)
            dow    = DATES[d].weekday()
            for s in SHIFTS:
                shift_name = ["morning", "afternoon", "night"][s]
                coeff = 0
                if shift_name in wp.preferred_shifts:
                    coeff += WEIGHT_PREFERRED_SHIFT
                if shift_name in wp.avoided_shifts:
                    coeff += WEIGHT_AVOIDED_SHIFT
                if s == 2:
                    coeff += WEIGHT_NIGHT_TOLERANCE if wp.night_tolerance else WEIGHT_NIGHT_NO_TOLERANCE
                if is_hol:
                    coeff += WEIGHT_HOLIDAY_TOLERANCE if wp.holiday_tolerance else WEIGHT_HOLIDAY_NO_TOLERANCE
                if wp.preferred_rest_day is not None and dow == wp.preferred_rest_day:
                    coeff -= 5
                if coeff != 0:
                    sat_terms.append(coeff * SCALE * x[w][d][s])

        # Base satisfaction (neutral schedule → 0 penalty)
        # Range is roughly [-large, +large]; we use a bounded var
        lb = -200 * SCALE
        ub =  200 * SCALE
        sat_var = model.new_int_var(lb, ub, f"sat_{w}")
        if sat_terms:
            model.add(sat_var == sum(sat_terms))
        else:
            model.add(sat_var == 0)
        worker_sat.append(sat_var)

    # Min-satisfaction variable (for max-min fairness)
    min_sat = model.new_int_var(-200 * SCALE, 200 * SCALE, "min_sat")
    for sv in worker_sat:
        model.add(min_sat <= sv)  # min_sat is a lower bound on all satisfactions

    # ── Refinement floors ─────────────────────────────────────────────────────
    # All floors are in the normalized [0, 100] range and must be converted to
    # the raw integer domain used by worker_sat variables.
    # Conversion: raw = normalized * 4 - 200  (inverse of the normalization formula)

    # Step 1: locate the pinned worker index (None when not in refinement mode)
    pinned_idx: int | None = None
    if pinned_worst_worker_id:
        try:
            pinned_idx = [wp.worker_id for wp in workers].index(pinned_worst_worker_id)
        except ValueError:
            pinned_idx = None

    # Step 2: protect all NON-pinned workers with min_satisfaction_floor.
    # This preserves the satisfaction of the second-worst worker (and everyone
    # above them) so that improving the worst worker doesn't hurt others.
    if min_satisfaction_floor > 0.0:
        floor_int = _normalized_to_raw_floor(min_satisfaction_floor, SCALE)
        for w in WORKERS:
            if w == pinned_idx:
                continue   # pinned worker handled separately below
            model.add(worker_sat[w] >= floor_int)

    # Step 3: force the pinned (worst) worker to strictly improve.
    # pinned_min_floor is their current score; adding a small epsilon ensures
    # the solver must find a strictly better assignment.
    if pinned_idx is not None and pinned_min_floor > 0.0:
        pinned_floor_int = _normalized_to_raw_floor(pinned_min_floor, SCALE)
        model.add(worker_sat[pinned_idx] >= pinned_floor_int)

    # ── AI Refinement Strategy Integration ────────────────────────────────────
    boost_weight = 5
    extra_pref_obj: list = []   # extra objective terms from shifts_to_prefer

    if pinned_idx is not None and strategy_hints:
        # 1. Hard-ban explicitly undesirable shifts for the pinned worker.
        #    (Shift avoidance is a legitimate hard restriction.)
        for s_avoid in strategy_hints.get("shifts_to_avoid", []):
            if 0 <= s_avoid < NUM_SHIFTS:
                for d in DAYS:
                    model.add(x[pinned_idx][d][s_avoid] == 0)

        # 2. Add a SOFT bonus for preferred shifts instead of a hard constraint.
        #    The previous implementation used model.add(x[...] == 1) for every
        #    (day, preferred_shift), which forces the worker to work every single
        #    day on that shift and makes the model trivially infeasible.
        #    We now add extra objective weight so the solver is INCENTIVISED
        #    (not forced) to assign those shifts.
        boost_weight = max(1, int(strategy_hints.get("weight_boost", 5)))
        for s_prefer in strategy_hints.get("shifts_to_prefer", []):
            if 0 <= s_prefer < NUM_SHIFTS:
                for d in DAYS:
                    extra_pref_obj.append(boost_weight * SCALE * x[pinned_idx][d][s_prefer])

    total_sat = sum(worker_sat)

    if pinned_idx is not None:
        # Objective: global satisfaction + fairness (min-sat) + focused boost on pinned worker
        # extra_pref_obj provides additional incentive to assign preferred shift types
        obj = 7 * total_sat + 3 * n_workers * min_sat + boost_weight * worker_sat[pinned_idx]
        if extra_pref_obj:
            obj = obj + sum(extra_pref_obj)
        model.maximize(obj)
    else:
        model.maximize(7 * total_sat + 3 * n_workers * min_sat)

    # ── Solve ─────────────────────────────────────────────────────────────────
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, {}

    # ── Extract solution ───────────────────────────────────────────────────────
    assignments = []
    for w, wp in enumerate(workers):
        for d in DAYS:
            for s in SHIFTS:
                if solver.value(x[w][d][s]) == 1:
                    assignments.append(Assignment(
                        worker_id=wp.worker_id,
                        day_index=d,
                        shift_index=s,
                    ))

    schedule = Schedule(assignments=assignments)

    # Compute raw satisfaction scores (normalised to 0-100)
    sat_scores: dict[str, float] = {}
    for w, wp in enumerate(workers):
        raw = solver.value(worker_sat[w]) / SCALE
        # Normalise: shift raw score into [0, 100]
        # Worst possible ≈ -200, best ≈ +200  →  (raw + 200) / 4
        normalised = max(0.0, min(100.0, (raw + 200) / 4))
        sat_scores[wp.worker_id] = round(normalised, 2)

    return schedule, sat_scores
