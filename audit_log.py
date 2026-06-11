"""
SmartScheduler – Audit Log
============================
Log immutabile (append-only) di ogni esecuzione del sistema.

Progettato per consentire il debugging post-mortem quando il sistema è in
modalità dinamica (Fase 1b). Ogni riga del file JSONL è autosufficiente:
contiene tutto il necessario per riprodurre l'esecuzione.

File di log: scheduling_audit.jsonl (nella directory del progetto)

Formato di ogni riga:
{
  "timestamp": "2026-12-07T10:23:45+01:00",
  "run_id": "uuid4",
  "use_case": "A" | "B",
  "mode": "STATIC" | "DYNAMIC",
  "classification": "known" | "mixed" | "refused",
  "extensions_applied": [{"type": "...", "parameters": {...}, "source_text": "..."}],
  "validation_passed": true | false,
  "validation_errors": [...],
  "outcome": "FEASIBLE" | "INFEASIBLE" | "REFUSED" | "ERROR",
  "num_workers": 10,
  "num_assignments": 250,
  "min_satisfaction": 48.5,
  "duration_seconds": 12.3,
  "notes": "..."
}
"""

from __future__ import annotations
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Posizione del file di log (stessa directory del progetto)
_DEFAULT_LOG_PATH = Path(__file__).parent / "scheduling_audit.jsonl"


def log_scheduling_run(
    use_case: str,
    classification: str,
    outcome: str,
    extensions_applied: Optional[list[dict]] = None,
    validation_passed: bool = True,
    validation_errors: Optional[list[str]] = None,
    num_workers: int = 0,
    num_assignments: int = 0,
    min_satisfaction: float = 0.0,
    duration_seconds: float = 0.0,
    notes: str = "",
    log_path: Optional[Path] = None,
) -> str:
    """
    Appende una riga JSON al file di audit log.

    Args:
        use_case:          "A" o "B".
        classification:    "known" | "mixed" | "refused".
        outcome:           "FEASIBLE" | "INFEASIBLE" | "REFUSED" | "ERROR".
        extensions_applied: Lista di dizionari ExtensionSpec serializzati.
        validation_passed: Se la validazione degli ExtensionSpec è passata.
        validation_errors: Lista di errori di validazione (se presenti).
        num_workers:       Numero di lavoratori nel run.
        num_assignments:   Numero di assegnazioni nel calendario prodotto.
        min_satisfaction:  Punteggio di soddisfazione minimo (0-100).
        duration_seconds:  Durata totale del run in secondi.
        notes:             Note libere per il debugging.
        log_path:          Path opzionale del file di log (default: scheduling_audit.jsonl).

    Returns:
        Il run_id UUID assegnato a questo run.
    """
    run_id = str(uuid.uuid4())
    mode = "DYNAMIC" if classification in ("mixed",) else "STATIC"

    record = {
        "timestamp":        datetime.now(tz=timezone.utc).isoformat(),
        "run_id":           run_id,
        "use_case":         use_case,
        "mode":             mode,
        "classification":   classification,
        "extensions_applied": extensions_applied or [],
        "validation_passed": validation_passed,
        "validation_errors": validation_errors or [],
        "outcome":          outcome,
        "num_workers":      num_workers,
        "num_assignments":  num_assignments,
        "min_satisfaction": min_satisfaction,
        "duration_seconds": round(duration_seconds, 3),
        "notes":            notes,
    }

    target = Path(log_path) if log_path else _DEFAULT_LOG_PATH

    try:
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        # Il log non deve mai bloccare il sistema principale
        print(f"  [AuditLog] AVVISO: impossibile scrivere nel log: {exc}")

    return run_id


def read_audit_log(
    log_path: Optional[Path] = None,
    last_n: Optional[int] = None,
    filter_mode: Optional[str] = None,
    filter_outcome: Optional[str] = None,
) -> list[dict]:
    """
    Legge il file di audit log e restituisce le voci come lista di dizionari.

    Args:
        log_path:       Path del file (default: scheduling_audit.jsonl).
        last_n:         Se specificato, restituisce solo le ultime N voci.
        filter_mode:    Filtra per 'STATIC' o 'DYNAMIC'.
        filter_outcome: Filtra per 'FEASIBLE', 'INFEASIBLE', 'REFUSED', 'ERROR'.

    Returns:
        Lista di dict, in ordine cronologico (più vecchio prima).
    """
    target = Path(log_path) if log_path else _DEFAULT_LOG_PATH

    if not target.exists():
        return []

    records = []
    with open(target, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # riga corrotta → salta

    # Filtri
    if filter_mode:
        records = [r for r in records if r.get("mode") == filter_mode]
    if filter_outcome:
        records = [r for r in records if r.get("outcome") == filter_outcome]

    if last_n is not None:
        records = records[-last_n:]

    return records


def get_dynamic_run_summary(log_path: Optional[Path] = None) -> str:
    """
    Restituisce un riassunto testuale di tutti i run in modalità DYNAMIC.
    Utile per il debugging e per il report al team di manutenzione.
    """
    dynamic_runs = read_audit_log(log_path=log_path, filter_mode="DYNAMIC")

    if not dynamic_runs:
        return "Nessun run in modalità DYNAMIC registrato nel log."

    lines = [f"=== Run in modalità DYNAMIC: {len(dynamic_runs)} totali ===\n"]
    for r in dynamic_runs:
        ext_types = [e.get("type", "?") for e in r.get("extensions_applied", [])]
        lines.append(
            f"[{r['timestamp'][:19]}] run_id={r['run_id'][:8]}… "
            f"use_case={r['use_case']} outcome={r['outcome']} "
            f"extensions={ext_types} "
            f"validation={'OK' if r['validation_passed'] else 'FALLITA'}"
        )

    return "\n".join(lines)
