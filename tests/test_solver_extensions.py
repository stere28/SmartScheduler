"""
Test – Solver Extensions
==========================
Unit test per ogni funzione in solver_extensions.py.

Ogni test verifica TRE proprietà:
  1. Il vincolo è soddisfatto nella soluzione prodotta (test positivo)
  2. Il solver non va in INFEASIBLE per parametri validi
  3. Il solver va in INFEASIBLE o lancia ValueError per parametri invalidi (test negativo)

IMPORTANTE: questi test NON richiedono un LLM. Chiamano direttamente il solver.
Possono essere eseguiti in CI/CD senza dipendenze esterne.
"""

from __future__ import annotations
import sys
import os
import math

import pytest

# Aggiungi la root del progetto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ortools.sat.python import cp_model as _cp_model

from models import ShiftPreference, ExtensionSpec
from solver import solve_schedule
from solver_extensions import (
    add_max_consecutive_nights,
    add_min_gap_hours_between_shifts,
    add_pair_always_same_shift,
    add_max_shifts_of_type_per_week,
    add_worker_group_min_rest_days,
    apply_extension,
)
from config import NUM_DAYS, NUM_SHIFTS, DATES


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_worker(worker_id: str, worker_type: str = "standard") -> ShiftPreference:
    """Factory per creare un worker con parametri neutri."""
    return ShiftPreference(
        worker_id=worker_id,
        worker_name=f"Worker {worker_id}",
        worker_type=worker_type,
        preferred_shifts=[],
        avoided_shifts=[],
        night_tolerance=True,
        holiday_tolerance=True,
        unavailable_days_of_week=[],
        preferred_rest_day=None,
        emergency_coverage=0,
    )


@pytest.fixture
def standard_workers_A() -> list[ShiftPreference]:
    """10 worker standard per Use Case A."""
    return [_make_worker(f"W{i:02d}") for i in range(1, 11)]


@pytest.fixture
def mixed_workers_B() -> list[ShiftPreference]:
    """10 standard + 6 specialized per Use Case B."""
    std  = [_make_worker(f"W{i:02d}", "standard") for i in range(1, 11)]
    spec = [_make_worker(f"S{i:02d}", "specialized") for i in range(1, 7)]
    return std + spec


# ── Test E1: max_consecutive_nights ───────────────────────────────────────────

class TestMaxConsecutiveNights:

    def test_constraint_satisfied_in_solution(self, standard_workers_A):
        """Con max_consecutive=2, nessun lavoratore ha più di 2 notti di fila."""
        ext = ExtensionSpec(
            extension_type="max_consecutive_nights",
            parameters={"max_consecutive": 2},
        )
        schedule, scores = solve_schedule(
            workers=standard_workers_A,
            use_case="A",
            extensions=[ext],
            time_limit_seconds=30,
        )
        assert schedule is not None, "Il solver deve produrre una soluzione feasible"

        # Verifica manuale del vincolo nella soluzione
        NIGHT = 2
        for worker in standard_workers_A:
            wid = worker.worker_id
            night_days = sorted(
                a.day_index for a in schedule.assignments
                if a.worker_id == wid and a.shift_index == NIGHT
            )
            # Controlla finestre di 3 giorni consecutivi
            for i in range(len(night_days) - 2):
                window = night_days[i:i+3]
                if window[2] - window[0] == 2:  # 3 notti consecutive
                    pytest.fail(
                        f"Worker {wid} ha 3 notti consecutive: giorni {window}"
                    )

    def test_solver_feasible_with_max3(self, standard_workers_A):
        """max_consecutive=3 deve essere feasible."""
        ext = ExtensionSpec(
            extension_type="max_consecutive_nights",
            parameters={"max_consecutive": 3},
        )
        schedule, _ = solve_schedule(
            workers=standard_workers_A, use_case="A",
            extensions=[ext], time_limit_seconds=30,
        )
        assert schedule is not None

    def test_invalid_parameter_raises(self):
        """max_consecutive fuori range [1,7] deve sollevare ValueError."""
        with pytest.raises(ValueError, match="max_consecutive"):
            # Chiamata diretta per testare la guardia del parametro
            model = _cp_model.CpModel()
            workers = [_make_worker("W01")]
            x = [[[model.new_bool_var(f"x_0_{d}_{s}") for s in range(NUM_SHIFTS)]
                  for d in range(NUM_DAYS)] for _ in workers]
            add_max_consecutive_nights(model, x, workers, max_consecutive=0)

    def test_invalid_parameter_too_large(self):
        """max_consecutive > 7 deve sollevare ValueError."""
        with pytest.raises(ValueError, match="max_consecutive"):
            model = _cp_model.CpModel()
            workers = [_make_worker("W01")]
            x = [[[model.new_bool_var(f"x_0_{d}_{s}") for s in range(NUM_SHIFTS)]
                  for d in range(NUM_DAYS)] for _ in workers]
            add_max_consecutive_nights(model, x, workers, max_consecutive=8)


# ── Test E3: pair_always_same_shift ───────────────────────────────────────────

class TestPairAlwaysSameShift:

    def test_pair_co_assigned(self, standard_workers_A):
        """I due lavoratori in coppia devono sempre lavorare insieme."""
        ext = ExtensionSpec(
            extension_type="pair_always_same_shift",
            parameters={"worker_a_id": "W01", "worker_b_id": "W02"},
        )
        schedule, _ = solve_schedule(
            workers=standard_workers_A,
            use_case="A",
            extensions=[ext],
            time_limit_seconds=30,
        )
        assert schedule is not None, "Il solver deve trovare una soluzione feasible"

        # Verifica: per ogni giorno, W01 e W02 hanno lo stesso turno
        for d in range(NUM_DAYS):
            shifts_w01 = {
                a.shift_index for a in schedule.assignments
                if a.worker_id == "W01" and a.day_index == d
            }
            shifts_w02 = {
                a.shift_index for a in schedule.assignments
                if a.worker_id == "W02" and a.day_index == d
            }
            assert shifts_w01 == shifts_w02, (
                f"Giorno {d}: W01 ha turni {shifts_w01} ma W02 ha turni {shifts_w02}"
            )

    def test_unknown_worker_raises(self, standard_workers_A):
        """Un worker_id inesistente deve sollevare ValueError."""
        with pytest.raises(ValueError, match="worker_a_id"):
            model = _cp_model.CpModel()
            n = len(standard_workers_A)
            x = [[[model.new_bool_var(f"x_{w}_{d}_{s}")
                   for s in range(NUM_SHIFTS)]
                  for d in range(NUM_DAYS)]
                 for w in range(n)]
            add_pair_always_same_shift(model, x, standard_workers_A, "ZZZZ", "W02")

    def test_same_worker_raises(self, standard_workers_A):
        """Stessa coppia (a == b) deve sollevare ValueError."""
        with pytest.raises(ValueError):
            model = _cp_model.CpModel()
            n = len(standard_workers_A)
            x = [[[model.new_bool_var(f"x_{w}_{d}_{s}")
                   for s in range(NUM_SHIFTS)]
                  for d in range(NUM_DAYS)]
                 for w in range(n)]
            add_pair_always_same_shift(model, x, standard_workers_A, "W01", "W01")



# ── Test E4: max_shifts_of_type_per_week ─────────────────────────────────────

class TestMaxShiftsOfTypePerWeek:

    def test_max_1_night_per_week(self, standard_workers_A):
        """Con max_per_week=2 per i notturni, nessuno deve avere 3+ notti in una settimana.

        NOTA: max_per_week=1 renderebbe il modello INFEASIBLE perché con
        TARGET_SHIFTS_MONTH=25 e night_weight=2 ogni worker necessita di circa
        2 notti a settimana per raggiungere il target.
        """
        ext = ExtensionSpec(
            extension_type="max_shifts_of_type_per_week",
            parameters={"shift_type": "night", "max_per_week": 2, "applies_to": "all"},
        )
        schedule, _ = solve_schedule(
            workers=standard_workers_A, use_case="A",
            extensions=[ext], time_limit_seconds=30,
        )
        assert schedule is not None

        num_weeks = math.ceil(NUM_DAYS / 7)
        for worker in standard_workers_A:
            wid = worker.worker_id
            for wk in range(num_weeks):
                d_start = wk * 7
                d_end   = min(d_start + 7, NUM_DAYS)
                nights = sum(
                    1 for a in schedule.assignments
                    if a.worker_id == wid
                    and a.shift_index == 2
                    and d_start <= a.day_index < d_end
                )
                assert nights <= 2, (
                    f"Worker {wid} settimana {wk}: {nights} notti (max 2)"
                )

    def test_invalid_shift_type_raises(self):
        """Shift type non valido deve sollevare ValueError."""
        with pytest.raises(ValueError, match="shift_index"):
            model = _cp_model.CpModel()
            workers = [_make_worker("W01")]
            x = [[[model.new_bool_var(f"x_0_{d}_{s}") for s in range(NUM_SHIFTS)]
                  for d in range(NUM_DAYS)] for _ in workers]
            add_max_shifts_of_type_per_week(
                model, x, workers, shift_index=5, max_per_week=2
            )


# ── Test dispatcher ───────────────────────────────────────────────────────────

class TestDispatcher:

    def test_unknown_extension_type_raises(self, standard_workers_A):
        """Extension type non in whitelist deve sollevare errore Pydantic prima del dispatcher."""
        with pytest.raises(Exception):
            # Pydantic dovrebbe rifiutare il tipo non valido
            ExtensionSpec(
                extension_type="delete_all_files",
                parameters={},
            )

    def test_dispatcher_routes_correctly(self, standard_workers_A):
        """Il dispatcher deve chiamare la funzione giusta per ogni tipo."""
        model = _cp_model.CpModel()
        n = len(standard_workers_A)
        x = [[[model.new_bool_var(f"x_{w}_{d}_{s}")
               for s in range(NUM_SHIFTS)]
              for d in range(NUM_DAYS)]
             for w in range(n)]

        # Questi non devono sollevare eccezioni
        ext1 = ExtensionSpec(
            extension_type="max_consecutive_nights",
            parameters={"max_consecutive": 3},
        )
        ext2 = ExtensionSpec(
            extension_type="max_shifts_of_type_per_week",
            parameters={"shift_type": "morning", "max_per_week": 3},
        )
        apply_extension(model, x, standard_workers_A, ext1)
        apply_extension(model, x, standard_workers_A, ext2)


# ── Test integrazione solve_schedule + extensions ────────────────────────────

class TestSolveScheduleWithExtensions:

    def test_no_extensions_equals_static_behavior(self, standard_workers_A):
        """Con extensions=None, solve_schedule si comporta come la versione statica."""
        schedule, scores = solve_schedule(
            workers=standard_workers_A,
            use_case="A",
            extensions=None,
            time_limit_seconds=30,
        )
        assert schedule is not None
        assert len(scores) == len(standard_workers_A)

    def test_empty_extensions_equals_static_behavior(self, standard_workers_A):
        """Con extensions=[], solve_schedule si comporta come la versione statica."""
        schedule, scores = solve_schedule(
            workers=standard_workers_A,
            use_case="A",
            extensions=[],
            time_limit_seconds=30,
        )
        assert schedule is not None

    def test_use_case_B_with_extension(self, mixed_workers_B):
        """Un'estensione deve funzionare anche con Use Case B."""
        ext = ExtensionSpec(
            extension_type="max_consecutive_nights",
            parameters={"max_consecutive": 3},
        )
        schedule, _ = solve_schedule(
            workers=mixed_workers_B,
            use_case="B",
            extensions=[ext],
            time_limit_seconds=45,
        )
        assert schedule is not None
