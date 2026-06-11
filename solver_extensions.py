"""
SmartScheduler – Solver Extensions
=====================================
Libreria di blocchi di vincoli pre-compilati per la Fase 1b (Generazione Dinamica).

GARANZIE DI SICUREZZA:
  ✓ Nessun eval() / exec() / subprocess / codice dinamico
  ✓ Ogni funzione ha firma Python tipizzata e documentazione degli invarianti
  ✓ Ogni funzione è coperta da unit test in tests/test_solver_extensions.py
  ✓ I vincoli H1-H7 del solver statico rimangono sempre attivi e non possono
    essere disabilitati o sovrascritti da queste estensioni

COME AGGIUNGERE UN NUOVO EXTENSION PATTERN:
  1. Scrivi la funzione `add_<nome>(model, x, workers, **params) -> None`
  2. Aggiungi il tipo in ExtensionSpec.extension_type (models.py)
  3. Aggiungi il descrittore in EXTENSION_PATTERNS (constraint_registry.py)
  4. Scrivi i test in tests/test_solver_extensions.py
  5. Aggiorna il dispatcher `apply_extension()` in questo file
  Solo dopo questi passi il vincolo è considerato "sicuro".
"""

from __future__ import annotations
import math
from typing import Optional

from ortools.sat.python import cp_model

from models import ExtensionSpec, ShiftPreference
from config import NUM_DAYS, NUM_SHIFTS, SHIFT_HOURS, DATES


# ── Dispatcher ─────────────────────────────────────────────────────────────────

def apply_extension(
    model: cp_model.CpModel,
    x: list,
    workers: list[ShiftPreference],
    ext: ExtensionSpec,
) -> None:
    """
    Dispatcher centrale: riceve un ExtensionSpec validato e chiama la funzione
    di vincolo appropriata. NON genera codice; usa esclusivamente dispatch statico.

    Args:
        model:   Il CpModel OR-Tools in costruzione.
        x:       Variabili decisionali x[w][d][s].
        workers: Lista degli ShiftPreference workers.
        ext:     L'ExtensionSpec validato (tipo + parametri).

    Raises:
        ValueError: se extension_type non è riconosciuto (non dovrebbe mai
                    accadere grazie alla validazione Pydantic, ma difesa in profondità).
    """
    p = ext.parameters

    if ext.extension_type == "max_consecutive_nights":
        add_max_consecutive_nights(
            model, x, workers,
            max_consecutive=int(p["max_consecutive"]),
        )

    elif ext.extension_type == "min_gap_hours_between_shifts":
        add_min_gap_hours_between_shifts(
            model, x, workers,
            min_gap_hours=int(p["min_gap_hours"]),
        )

    elif ext.extension_type == "pair_always_same_shift":
        add_pair_always_same_shift(
            model, x, workers,
            worker_a_id=str(p["worker_a_id"]),
            worker_b_id=str(p["worker_b_id"]),
        )

    elif ext.extension_type == "max_shifts_of_type_per_week":
        shift_name_to_idx = {"morning": 0, "afternoon": 1, "night": 2}
        add_max_shifts_of_type_per_week(
            model, x, workers,
            shift_index=shift_name_to_idx[str(p["shift_type"])],
            max_per_week=int(p["max_per_week"]),
            applies_to=p.get("applies_to", "all"),
        )

    elif ext.extension_type == "worker_group_min_rest_days":
        add_worker_group_min_rest_days(
            model, x, workers,
            worker_ids=list(p["worker_ids"]),
            min_rest_days=int(p["min_rest_days"]),
            day_range=p.get("in_day_range", None),
        )

    else:
        # Difesa in profondità: non dovrebbe mai arrivare qui grazie a Pydantic
        raise ValueError(
            f"Extension type non riconosciuto: '{ext.extension_type}'. "
            "Aggiornare il dispatcher in solver_extensions.py."
        )


# ── E1: Massimo N notti consecutive ───────────────────────────────────────────

def add_max_consecutive_nights(
    model: cp_model.CpModel,
    x: list,
    workers: list[ShiftPreference],
    max_consecutive: int,
) -> None:
    """
    [E1] Vincolo: al massimo `max_consecutive` turni notturni consecutivi
    per ogni lavoratore.

    Invariante matematico: per ogni lavoratore w e per ogni finestra di
    (max_consecutive + 1) giorni, la somma dei turni notturni è ≤ max_consecutive.

    Args:
        max_consecutive: intero in [1, 7]. Valore consigliato: 3.

    Test di riferimento: tests/test_solver_extensions.py::test_max_consecutive_nights
    """
    if not (1 <= max_consecutive <= 7):
        raise ValueError(
            f"max_consecutive deve essere tra 1 e 7, ricevuto: {max_consecutive}"
        )

    NIGHT_SHIFT = 2
    n_workers = len(workers)
    window = max_consecutive + 1  # una finestra di size (N+1) non può avere N+1 notti

    for w in range(n_workers):
        for d_start in range(NUM_DAYS - max_consecutive):
            # In qualsiasi finestra di (max_consecutive + 1) giorni, al massimo
            # max_consecutive notti
            night_vars_in_window = [
                x[w][d][NIGHT_SHIFT]
                for d in range(d_start, min(d_start + window, NUM_DAYS))
            ]
            model.add(sum(night_vars_in_window) <= max_consecutive)


# ── E2: Pausa minima in ore tra turni ─────────────────────────────────────────

def add_min_gap_hours_between_shifts(
    model: cp_model.CpModel,
    x: list,
    workers: list[ShiftPreference],
    min_gap_hours: int,
) -> None:
    """
    [E2] Vincolo: garantisce almeno `min_gap_hours` ore di pausa tra la fine
    di un turno e l'inizio del successivo per lo stesso lavoratore.

    Schema degli orari:
      morning   08:00–14:00  (ora_inizio=8,  durata=6h)
      afternoon 14:00–20:00  (ora_inizio=14, durata=6h)
      night     20:00–08:00  (ora_inizio=20, durata=12h, termina giorno+1)

    La logica del vincolo:
      - Per ogni coppia (turno_A su giorno D, turno_B su giorno D' ≥ D),
        se fine(A) + min_gap > inizio(B) → non possono coexistere.
      - Implementato come add_implication per efficienza.

    Args:
        min_gap_hours: intero in [8, 24].

    Test di riferimento: tests/test_solver_extensions.py::test_min_gap_hours
    """
    if not (8 <= min_gap_hours <= 24):
        raise ValueError(
            f"min_gap_hours deve essere tra 8 e 24, ricevuto: {min_gap_hours}"
        )

    # Ora di inizio e fine di ogni turno in ore dall'inizio del giorno
    # (notte termina il giorno successivo a ora 8 → fine = 20 + 12 = 32)
    shift_start = [8, 14, 20]
    shift_end   = [14, 20, 32]  # 32 = 08:00 del giorno successivo

    n_workers = len(workers)
    SHIFTS = range(NUM_SHIFTS)

    for w in range(n_workers):
        for d in range(NUM_DAYS):
            for s in SHIFTS:
                end_hour = shift_end[s]
                # Turno s finisce a ora end_hour (relativa al giorno d)
                # Consideriamo i turni nei giorni successivi entro la finestra
                for d2 in range(d, min(d + 3, NUM_DAYS)):  # max 2 giorni avanti
                    day_offset_hours = (d2 - d) * 24
                    for s2 in SHIFTS:
                        start_hour2 = shift_start[s2] + day_offset_hours
                        # Se non c'è sufficiente pausa tra la fine di (d,s) e l'inizio di (d2,s2)
                        if d2 == d and s2 <= s:
                            continue  # stessa coppia o già coperta da H2
                        gap = start_hour2 - end_hour
                        if 0 < gap < min_gap_hours:
                            # Impossibile avere entrambi
                            model.add_implication(x[w][d][s], x[w][d2][s2].negated())


# ── E3: Due lavoratori sempre sullo stesso turno ───────────────────────────────

def add_pair_always_same_shift(
    model: cp_model.CpModel,
    x: list,
    workers: list[ShiftPreference],
    worker_a_id: str,
    worker_b_id: str,
) -> None:
    """
    [E3] Vincolo: due lavoratori specifici devono essere assegnati allo stesso
    turno ogni giorno (o entrambi liberi).

    Formalmente: per ogni giorno D e turno S,
      x[a][D][S] = x[b][D][S]

    Utile per coppie tutor-studente o lavoratori con complementarità clinica.

    Args:
        worker_a_id: worker_id del primo lavoratore.
        worker_b_id: worker_id del secondo lavoratore.

    Raises:
        ValueError: se uno dei worker_id non esiste nella lista workers.

    Test di riferimento: tests/test_solver_extensions.py::test_pair_always_same_shift
    """
    id_to_idx = {w.worker_id: i for i, w in enumerate(workers)}

    if worker_a_id not in id_to_idx:
        raise ValueError(f"worker_a_id '{worker_a_id}' non trovato nella lista workers.")
    if worker_b_id not in id_to_idx:
        raise ValueError(f"worker_b_id '{worker_b_id}' non trovato nella lista workers.")
    if worker_a_id == worker_b_id:
        raise ValueError(
            f"worker_a_id e worker_b_id non possono essere lo stesso lavoratore ('{worker_a_id}')."
        )

    a = id_to_idx[worker_a_id]
    b = id_to_idx[worker_b_id]

    SHIFTS = range(NUM_SHIFTS)

    for d in range(NUM_DAYS):
        for s in SHIFTS:
            # x[a][d][s] == x[b][d][s]  →  entrambi lavorano o entrambi sono liberi
            model.add(x[a][d][s] == x[b][d][s])


# ── E4: Massimo N turni di un tipo per settimana ───────────────────────────────

def add_max_shifts_of_type_per_week(
    model: cp_model.CpModel,
    x: list,
    workers: list[ShiftPreference],
    shift_index: int,
    max_per_week: int,
    applies_to: object = "all",
) -> None:
    """
    [E4] Vincolo: al massimo `max_per_week` turni di tipo `shift_index` per
    settimana per i lavoratori specificati.

    Args:
        shift_index:  0=morning, 1=afternoon, 2=night.
        max_per_week: intero in [0, 7].
        applies_to:   'all' oppure lista di worker_id.

    Test di riferimento: tests/test_solver_extensions.py::test_max_shifts_of_type_per_week
    """
    if not (0 <= max_per_week <= 7):
        raise ValueError(f"max_per_week deve essere tra 0 e 7, ricevuto: {max_per_week}")
    if shift_index not in (0, 1, 2):
        raise ValueError(f"shift_index deve essere 0, 1 o 2, ricevuto: {shift_index}")

    n_workers = len(workers)
    num_weeks = math.ceil(NUM_DAYS / 7)

    # Determina quali workers applicare
    if applies_to == "all":
        target_workers = list(range(n_workers))
    else:
        id_to_idx = {w.worker_id: i for i, w in enumerate(workers)}
        target_workers = [id_to_idx[wid] for wid in applies_to if wid in id_to_idx]

    for w in target_workers:
        for wk in range(num_weeks):
            day_start = wk * 7
            day_end   = min(day_start + 7, NUM_DAYS)
            shifts_in_week = [
                x[w][d][shift_index]
                for d in range(day_start, day_end)
            ]
            model.add(sum(shifts_in_week) <= max_per_week)


# ── E5: Giorni di riposo minimi per gruppo ────────────────────────────────────

def add_worker_group_min_rest_days(
    model: cp_model.CpModel,
    x: list,
    workers: list[ShiftPreference],
    worker_ids: list[str],
    min_rest_days: int,
    day_range: Optional[list[int]] = None,
) -> None:
    """
    [E5] Vincolo: ogni lavoratore in `worker_ids` deve avere almeno
    `min_rest_days` giorni completamente liberi nell'intervallo `day_range`.

    Un giorno è "libero" se il lavoratore non è assegnato ad alcun turno.

    Args:
        worker_ids:    Lista di worker_id a cui applicare il vincolo.
        min_rest_days: Intero in [1, 7].
        day_range:     Opzionale [day_start, day_end] (0-based, inclusivi).
                       Se None, applica sull'intero orizzonte.

    Test di riferimento: tests/test_solver_extensions.py::test_worker_group_min_rest_days
    """
    if not (1 <= min_rest_days <= 14):
        raise ValueError(
            f"min_rest_days deve essere tra 1 e 14, ricevuto: {min_rest_days}"
        )

    id_to_idx = {w.worker_id: i for i, w in enumerate(workers)}
    SHIFTS = range(NUM_SHIFTS)

    if day_range is not None:
        d_start, d_end = int(day_range[0]), int(day_range[1])
        d_start = max(0, d_start)
        d_end   = min(NUM_DAYS - 1, d_end)
    else:
        d_start, d_end = 0, NUM_DAYS - 1

    day_count = d_end - d_start + 1

    for wid in worker_ids:
        if wid not in id_to_idx:
            continue  # skip unknown workers gracefully
        w = id_to_idx[wid]

        # is_working[d] = 1 se il lavoratore lavora nel giorno d
        is_working = [
            sum(x[w][d][s] for s in SHIFTS)
            for d in range(d_start, d_end + 1)
        ]

        # Numero di giorni liberi = day_count - sum(is_working)
        # Dobbiamo avere: day_count - sum(is_working) >= min_rest_days
        # Equivalente: sum(is_working) <= day_count - min_rest_days
        max_working_days = day_count - min_rest_days
        if max_working_days < 0:
            # Vincolo impossibile per l'intervallo dato – segnalare ma non crashare
            import warnings
            warnings.warn(
                f"[E5] Worker '{wid}': min_rest_days={min_rest_days} > "
                f"day_count={day_count} nell'intervallo [{d_start},{d_end}]. "
                "Il vincolo potrebbe rendere il problema infeasible.",
                RuntimeWarning,
            )
        model.add(sum(is_working) <= max(0, max_working_days))
