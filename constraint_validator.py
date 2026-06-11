"""
SmartScheduler – Constraint Validator
========================================
Validatore strutturale che viene eseguito PRIMA che qualsiasi ExtensionSpec
raggiunga il solver. È la "guardia di sicurezza" della Fase 1b.

Responsabilità:
  1. Verifica che i parametri di ogni ExtensionSpec siano nei range sicuri
  2. Verifica che i worker_id referenziati esistano nella lista workers
  3. Verifica la compatibilità tra estensioni (es. due E3 sullo stesso lavoratore)
  4. Produce un ValidationReport con errori dettagliati, senza mai eseguire codice

Design: tutte le funzioni sono pure (nessun side effect) e restituiscono
ValidationResult immutabili. Non sollevano eccezioni, ma accumulano errori.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from models import ExtensionSpec, ShiftPreference
from constraint_registry import EXTENSION_PATTERNS
from config import NUM_DAYS


# ── Risultato di validazione ──────────────────────────────────────────────────

@dataclass(frozen=True)
class ValidationResult:
    """Risultato immutabile della validazione di un singolo ExtensionSpec."""
    passed: bool
    extension_type: str
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        status = "✓ OK" if self.passed else "✗ FALLITA"
        lines = [f"[{self.extension_type}] {status}"]
        for e in self.errors:
            lines.append(f"  ERRORE: {e}")
        for w in self.warnings:
            lines.append(f"  AVVISO: {w}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ValidationReport:
    """Report aggregato per una lista di ExtensionSpec."""
    all_passed: bool
    results: tuple[ValidationResult, ...]

    def summary(self) -> str:
        status = "✓ TUTTE LE VALIDAZIONI PASSATE" if self.all_passed else "✗ VALIDAZIONE FALLITA"
        lines = [f"\n=== Constraint Validation Report: {status} ==="]
        for r in self.results:
            lines.append(r.summary())
        return "\n".join(lines)

    @property
    def errors(self) -> list[str]:
        return [e for r in self.results for e in r.errors]


# ── Validatori per tipo ────────────────────────────────────────────────────────

def _validate_max_consecutive_nights(p: dict) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    if "max_consecutive" not in p:
        errors.append("Parametro obbligatorio 'max_consecutive' mancante.")
        return errors, warnings
    try:
        v = int(p["max_consecutive"])
    except (TypeError, ValueError):
        errors.append(f"'max_consecutive' deve essere un intero, ricevuto: {p['max_consecutive']!r}")
        return errors, warnings
    if not (1 <= v <= 7):
        errors.append(f"'max_consecutive' deve essere tra 1 e 7, ricevuto: {v}.")
    if v == 1:
        warnings.append("max_consecutive=1 significa mai 2 notti di fila: verificare la fattibilità con il team.")
    return errors, warnings


def _validate_min_gap_hours(p: dict) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    if "min_gap_hours" not in p:
        errors.append("Parametro obbligatorio 'min_gap_hours' mancante.")
        return errors, warnings
    try:
        v = int(p["min_gap_hours"])
    except (TypeError, ValueError):
        errors.append(f"'min_gap_hours' deve essere un intero, ricevuto: {p['min_gap_hours']!r}")
        return errors, warnings
    if not (8 <= v <= 24):
        errors.append(f"'min_gap_hours' deve essere tra 8 e 24, ricevuto: {v}.")
    if v > 12:
        warnings.append(f"min_gap_hours={v}h è molto restrittivo: potrebbe rendere infeasible il modello.")
    return errors, warnings


def _validate_pair_always_same_shift(
    p: dict, workers: list[ShiftPreference]
) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    worker_ids = {w.worker_id for w in workers}

    for key in ("worker_a_id", "worker_b_id"):
        if key not in p:
            errors.append(f"Parametro obbligatorio '{key}' mancante.")
        elif str(p[key]) not in worker_ids:
            errors.append(f"'{key}' = '{p[key]}' non corrisponde a nessun worker_id esistente.")

    if not errors and p.get("worker_a_id") == p.get("worker_b_id"):
        errors.append("'worker_a_id' e 'worker_b_id' non possono essere lo stesso lavoratore.")

    warnings.append(
        "Il vincolo E3 (pair_always_same_shift) è MEDIUM risk: potrebbe aumentare "
        "significativamente lo spazio di ricerca del solver."
    )
    return errors, warnings


def _validate_max_shifts_of_type_per_week(
    p: dict, workers: list[ShiftPreference]
) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    valid_shift_types = {"morning", "afternoon", "night"}

    if "shift_type" not in p:
        errors.append("Parametro obbligatorio 'shift_type' mancante.")
    elif str(p["shift_type"]) not in valid_shift_types:
        errors.append(f"'shift_type' deve essere uno tra {valid_shift_types}, ricevuto: {p['shift_type']!r}")

    if "max_per_week" not in p:
        errors.append("Parametro obbligatorio 'max_per_week' mancante.")
    else:
        try:
            v = int(p["max_per_week"])
            if not (0 <= v <= 7):
                errors.append(f"'max_per_week' deve essere tra 0 e 7, ricevuto: {v}.")
            if v == 0:
                warnings.append("max_per_week=0 proibisce completamente questo tipo di turno al worker.")
        except (TypeError, ValueError):
            errors.append(f"'max_per_week' deve essere un intero, ricevuto: {p['max_per_week']!r}")

    applies_to = p.get("applies_to", "all")
    if applies_to != "all":
        worker_ids = {w.worker_id for w in workers}
        if isinstance(applies_to, list):
            unknown = [wid for wid in applies_to if wid not in worker_ids]
            if unknown:
                errors.append(f"applies_to contiene worker_id non esistenti: {unknown}")
        else:
            errors.append(f"applies_to deve essere 'all' o una lista di worker_id, ricevuto: {applies_to!r}")

    return errors, warnings


def _validate_worker_group_min_rest_days(
    p: dict, workers: list[ShiftPreference]
) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    worker_ids_set = {w.worker_id for w in workers}

    if "worker_ids" not in p or not isinstance(p["worker_ids"], list):
        errors.append("Parametro obbligatorio 'worker_ids' (lista) mancante.")
    else:
        unknown = [wid for wid in p["worker_ids"] if wid not in worker_ids_set]
        if unknown:
            warnings.append(f"worker_ids contiene ID sconosciuti (ignorati): {unknown}")

    if "min_rest_days" not in p:
        errors.append("Parametro obbligatorio 'min_rest_days' mancante.")
    else:
        try:
            v = int(p["min_rest_days"])
            if not (1 <= v <= 14):
                errors.append(f"'min_rest_days' deve essere tra 1 e 14, ricevuto: {v}.")
        except (TypeError, ValueError):
            errors.append(f"'min_rest_days' deve essere un intero, ricevuto: {p['min_rest_days']!r}")

    day_range = p.get("in_day_range")
    if day_range is not None:
        if not (isinstance(day_range, list) and len(day_range) == 2):
            errors.append("'in_day_range' deve essere una lista [day_start, day_end].")
        else:
            d0, d1 = day_range
            if not (0 <= int(d0) < int(d1) < NUM_DAYS):
                errors.append(
                    f"'in_day_range' [{d0}, {d1}] non valido per l'orizzonte di {NUM_DAYS} giorni."
                )

    return errors, warnings


# ── Validatore aggregato ───────────────────────────────────────────────────────

_VALIDATORS = {
    "max_consecutive_nights": lambda p, w: _validate_max_consecutive_nights(p),
    "min_gap_hours_between_shifts": lambda p, w: _validate_min_gap_hours(p),
    "pair_always_same_shift": _validate_pair_always_same_shift,
    "max_shifts_of_type_per_week": _validate_max_shifts_of_type_per_week,
    "worker_group_min_rest_days": _validate_worker_group_min_rest_days,
}


def validate_extension(
    ext: ExtensionSpec,
    workers: list[ShiftPreference],
) -> ValidationResult:
    """
    Valida un singolo ExtensionSpec contro la lista di workers corrente.

    Returns:
        ValidationResult con passed=True se e solo se non ci sono errori.
    """
    validator = _VALIDATORS.get(ext.extension_type)
    if validator is None:
        return ValidationResult(
            passed=False,
            extension_type=ext.extension_type,
            errors=(f"Tipo di estensione '{ext.extension_type}' non ha un validatore registrato.",),
        )

    errors, warnings = validator(ext.parameters, workers)
    return ValidationResult(
        passed=len(errors) == 0,
        extension_type=ext.extension_type,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def validate_all_extensions(
    extensions: list[ExtensionSpec],
    workers: list[ShiftPreference],
) -> ValidationReport:
    """
    Valida una lista di ExtensionSpec, producendo un ValidationReport aggregato.

    Questo è il punto di ingresso principale usato da agents.py prima di
    passare le estensioni al solver.

    Returns:
        ValidationReport con all_passed=True solo se TUTTE le validazioni passano.
    """
    results = tuple(validate_extension(ext, workers) for ext in extensions)
    return ValidationReport(
        all_passed=all(r.passed for r in results),
        results=results,
    )


def check_extension_compatibility(
    extensions: list[ExtensionSpec],
) -> list[str]:
    """
    Verifica la compatibilità tra estensioni (cross-validation).

    Restituisce una lista di avvisi (non errori bloccanti) su potenziali
    conflitti tra estensioni diverse.

    Esempio: due estensioni E3 (pair_always_same_shift) che formano un ciclo
    potrebbero creare un vincolo impossibile.
    """
    warnings: list[str] = []

    # Controllo: più di un E3 sullo stesso worker → potenziale ciclo
    pair_workers: list[str] = []
    for ext in extensions:
        if ext.extension_type == "pair_always_same_shift":
            p = ext.parameters
            pair_workers.extend([str(p.get("worker_a_id", "")), str(p.get("worker_b_id", ""))])

    duplicates = {wid for wid in pair_workers if pair_workers.count(wid) > 1}
    if duplicates:
        warnings.append(
            f"Worker {duplicates} appaiono in più vincoli pair_always_same_shift. "
            "Verificare che non si crei un ciclo di vincoli impossibile."
        )

    # Controllo: E1 (max_consecutive_nights) + H4 (2 gg riposo dopo notte)
    # Se max_consecutive=1 e FREE_DAYS_AFTER_NIGHT=2, i giorni notturni sono
    # molto limitati → avvisare sulla fattibilità
    for ext in extensions:
        if ext.extension_type == "max_consecutive_nights":
            v = int(ext.parameters.get("max_consecutive", 3))
            if v == 1:
                warnings.append(
                    "max_consecutive_nights=1 combinato con FREE_DAYS_AFTER_NIGHT=2 "
                    "significa che dopo ogni notte il worker non può lavorare per 2 giorni. "
                    "Verificare che il TARGET_SHIFTS_MONTH sia raggiungibile."
                )

    return warnings
