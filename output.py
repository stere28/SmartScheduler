"""
SmartScheduler – Output Formatter & Report Generator
======================================================
Pretty-prints schedules, verification reports, and exports CSV / JSON.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path
from datetime import date

from tabulate import tabulate
from colorama import Fore, Style, init as colorama_init

from config import SHIFT_NAMES, DATES, SHIFT_HOURS, SHIFT_WEIGHT
from models import SmartSchedulerState, Schedule, VerificationReport, ShiftPreference
from solver import _is_holiday

colorama_init(autoreset=True)


# ── Calendar view ──────────────────────────────────────────────────────────────

def print_schedule(state: SmartSchedulerState) -> None:
    """Print a compact per-worker monthly calendar."""
    if state.schedule is None:
        print("No schedule available.")
        return

    workers   = state.preferences.workers
    schedule  = state.schedule

    print(f"\n{'='*70}")
    print(f"  SmartScheduler – Use Case {state.use_case}")
    print(f"  Scheduling horizon: {DATES[0]} → {DATES[-1]}")
    print(f"{'='*70}\n")

    # Per-worker calendar
    SHIFT_ABBR = {"morning": "M", "afternoon": "A", "night": "N", "": "-"}
    header = ["Worker"] + [str(d.day) for d in DATES]
    rows = []
    for wp in workers:
        row = [f"{wp.worker_id} ({wp.worker_name})"]
        for d_idx, d in enumerate(DATES):
            assigned = [
                a for a in schedule.assignments
                if a.worker_id == wp.worker_id and a.day_index == d_idx
            ]
            cell = SHIFT_ABBR.get(assigned[0].shift_name, "-") if assigned else "-"
            if _is_holiday(d_idx):
                cell = f"[{cell}]"
            row.append(cell)
        rows.append(row)

    print(tabulate(rows, headers=header, tablefmt="rounded_grid"))

    # Per-shift staffing summary
    print(f"\n{'─'*70}")
    print("  Daily Staffing Summary (M=morning, A=afternoon, N=night)\n")
    staff_rows = []
    for d_idx, d in enumerate(DATES):
        day_str = f"{d.strftime('%a %d %b')}" + (" 🎄" if _is_holiday(d_idx) else "")
        for s_idx, s_name in enumerate(SHIFT_NAMES):
            staff = [
                a.worker_id for a in schedule.assignments
                if a.day_index == d_idx and a.shift_index == s_idx
            ]
            staff_rows.append([day_str, s_name.capitalize(), len(staff), ", ".join(staff)])

    print(tabulate(staff_rows, headers=["Date", "Shift", "# Workers", "Workers"],
                   tablefmt="simple"))


# ── Worker summary statistics ──────────────────────────────────────────────────

def print_worker_stats(state: SmartSchedulerState) -> None:
    """Print per-worker shift counts and hours."""
    if state.schedule is None:
        return

    workers  = state.preferences.workers
    schedule = state.schedule

    print(f"\n{'─'*70}")
    print("  Worker Statistics\n")
    rows = []
    for wp in workers:
        assignments = [a for a in schedule.assignments if a.worker_id == wp.worker_id]
        counts = [0, 0, 0]  # morning, afternoon, night
        for a in assignments:
            counts[a.shift_index] += 1
        total_hours = sum(SHIFT_HOURS[a.shift_index] for a in assignments)
        total_weight = sum(SHIFT_WEIGHT[a.shift_index] for a in assignments)
        night_hol = sum(
            1 for a in assignments if a.shift_index == 2 and _is_holiday(a.day_index)
        )
        rows.append([
            wp.worker_id, wp.worker_type,
            counts[0], counts[1], counts[2],
            total_weight, total_hours, night_hol,
        ])

    print(tabulate(
        rows,
        headers=["Worker", "Type", "Mrn", "Aft", "Ngt", "Workload", "Hours", "NightHol"],
        tablefmt="simple",
    ))


# ── Export functions ───────────────────────────────────────────────────────────

def export_csv(state: SmartSchedulerState, path: str = "schedule.csv") -> None:
    """Export schedule to CSV."""
    if state.schedule is None:
        return
    p = Path(path)
    with p.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["worker_id", "worker_name", "date", "shift", "hours", "is_holiday"])
        workers_map = {wp.worker_id: wp.worker_name for wp in state.preferences.workers}
        for a in sorted(state.schedule.assignments, key=lambda x: (x.worker_id, x.day_index)):
            d = DATES[a.day_index]
            writer.writerow([
                a.worker_id,
                workers_map.get(a.worker_id, ""),
                d.isoformat(),
                SHIFT_NAMES[a.shift_index],
                SHIFT_HOURS[a.shift_index],
                _is_holiday(a.day_index),
            ])
    print(f"\n  Schedule exported to {Fore.CYAN}{p.resolve()}{Style.RESET_ALL}")


def export_json(state: SmartSchedulerState, path: str = "schedule.json") -> None:
    """Export full state (schedule + verification) to JSON."""
    if state.schedule is None:
        return
    p = Path(path)
    data = {
        "use_case": state.use_case,
        "horizon": {"start": str(DATES[0]), "end": str(DATES[-1])},
        "assignments": [
            {
                "worker_id": a.worker_id,
                "date": str(DATES[a.day_index]),
                "shift": SHIFT_NAMES[a.shift_index],
                "hours": SHIFT_HOURS[a.shift_index],
                "is_holiday": _is_holiday(a.day_index),
            }
            for a in sorted(state.schedule.assignments, key=lambda x: (x.worker_id, x.day_index))
        ],
        "pipeline_log": state.history,
    }
    with p.open("w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Schedule exported to {Fore.CYAN}{p.resolve()}{Style.RESET_ALL}")
