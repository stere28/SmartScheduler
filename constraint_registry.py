"""
SmartScheduler – Constraint Registry
======================================
Catalogo dichiarativo di tutti i vincoli supportati dal solver statico ("Casi Noti").

Questo registro ha tre ruoli:
1. WHITELIST FORMALE: definisce esattamente quali vincoli sono considerati "noti" e sicuri.
2. PROMPT ENGINEERING: il suo contenuto viene iniettato nel system prompt del
   constraint_classifier_agent (Fase 1a) in modo che l'LLM sappia esattamente
   cosa può e cosa non può mappare.
3. DOCUMENTAZIONE VIVENTE: ogni ingegnere può leggere qui quali vincoli sono
   coperti da unit test nel solver statico.

IMPORTANTE: Per aggiungere un nuovo vincolo "noto", occorre:
  a) Aggiungere la sua definizione in KNOWN_CONSTRAINTS
  b) Implementare il mapping in solver.py o solver_extensions.py
  c) Scrivere un unit test in tests/
  d) Solo allora il vincolo è considerato "sicuro"
"""

from __future__ import annotations
from typing import Any


# ── Definizione dei Casi Noti ─────────────────────────────────────────────────

KNOWN_CONSTRAINTS: dict[str, dict[str, Any]] = {

    # ── HARD CONSTRAINTS (H1-H7) ──────────────────────────────────────────────

    "min_staff_per_shift": {
        "label": "H1 – Staffing minimo per turno",
        "description": (
            "Ogni turno (mattina, pomeriggio, notte) deve essere coperto da un "
            "numero minimo di lavoratori. Nel Caso A: ≥2 standard. Nel Caso B: "
            "≥1 specializzato, ≥1 standard, ≥3 totali."
        ),
        "parameters": {
            "use_case": "A o B",
            "min_standard_per_shift": "intero ≥ 0",
            "min_specialized_per_shift": "intero ≥ 0",
            "min_total_per_shift": "intero ≥ 0",
        },
        "solver_handler": "H1 in solver.py",
        "unit_tested": True,
    },

    "max_one_shift_per_day": {
        "label": "H2 – Al più 1 turno al giorno per lavoratore",
        "description": (
            "Un lavoratore non può essere assegnato a più di un turno nello stesso "
            "giorno."
        ),
        "parameters": {},
        "solver_handler": "H2 in solver.py",
        "unit_tested": True,
    },

    "no_night_to_morning_consecutive": {
        "label": "H3 – No turno consecutivo notte→mattina cross-day",
        "description": (
            "Se un lavoratore copre il turno di notte del giorno D, non può coprire "
            "il turno di mattina del giorno D+1 (il turno di notte termina alle 08:00)."
        ),
        "parameters": {},
        "solver_handler": "H3 in solver.py",
        "unit_tested": True,
    },

    "night_rest_days": {
        "label": "H4 – Giorni di riposo obbligatori dopo turno notturno",
        "description": (
            "Dopo ogni turno notturno, il lavoratore deve avere un numero minimo di "
            "giorni completamente liberi. Il valore di default è 2."
        ),
        "parameters": {
            "free_days_after_night": "intero (default: 2)",
        },
        "solver_handler": "H4 in solver.py",
        "unit_tested": True,
    },

    "workload_target": {
        "label": "H5 – Target di workload mensile",
        "description": (
            "Ogni lavoratore deve coprire esattamente TARGET_SHIFTS_MONTH unità di "
            "workload nel mese. I turni notturni contano 2 unità, gli altri 1."
        ),
        "parameters": {
            "target_shifts_month": "intero (default: 25)",
            "shift_weights": {"morning": 1, "afternoon": 1, "night": 2},
        },
        "solver_handler": "H5 in solver.py",
        "unit_tested": True,
    },

    "weekly_hours_cap": {
        "label": "H6 – Ore settimanali ≤ MAX_HOURS_PER_WEEK",
        "description": (
            "Le ore lavorate da un lavoratore in qualsiasi settimana del periodo "
            "non possono superare il massimo settimanale. Default: 36 ore."
        ),
        "parameters": {
            "max_hours_per_week": "intero (default: 36)",
            "shift_hours": {"morning": 6, "afternoon": 6, "night": 12},
        },
        "solver_handler": "H6 in solver.py",
        "unit_tested": True,
    },

    "worker_unavailability": {
        "label": "H7 – Indisponibilità per giorno della settimana",
        "description": (
            "Un lavoratore può dichiarare certi giorni della settimana come "
            "completamente non disponibili (es. 'non disponibile la domenica'). "
            "Mappato dal campo unavailable_days_of_week (0=Lun, 6=Dom)."
        ),
        "parameters": {
            "unavailable_days_of_week": "lista di interi 0-6",
        },
        "solver_handler": "H7 in solver.py",
        "unit_tested": True,
    },

    # ── SOFT CONSTRAINTS ──────────────────────────────────────────────────────

    "preferred_shift_bonus": {
        "label": "S1 – Bonus per turno preferito",
        "description": (
            "Se un lavoratore dichiara una preferenza per un tipo di turno "
            "(morning/afternoon/night), ogni assegnazione a quel turno contribuisce "
            "positivamente al suo punteggio di soddisfazione."
        ),
        "parameters": {
            "preferred_shifts": "lista di 'morning', 'afternoon', 'night'",
            "weight": "WEIGHT_PREFERRED_SHIFT (default: +10)",
        },
        "solver_handler": "soft terms in solver.py",
        "unit_tested": True,
    },

    "avoided_shift_penalty": {
        "label": "S2 – Penalità per turno evitato",
        "description": (
            "Se un lavoratore vuole evitare un certo turno, ogni assegnazione "
            "a quel turno penalizza il suo punteggio."
        ),
        "parameters": {
            "avoided_shifts": "lista di 'morning', 'afternoon', 'night'",
            "weight": "WEIGHT_AVOIDED_SHIFT (default: -15)",
        },
        "solver_handler": "soft terms in solver.py",
        "unit_tested": True,
    },

    "night_tolerance": {
        "label": "S3 – Tolleranza ai turni notturni",
        "description": (
            "Lavoratori con night_tolerance=True ricevono un bonus per ogni turno "
            "notturno assegnato. Quelli con night_tolerance=False ricevono una penalità."
        ),
        "parameters": {
            "night_tolerance": "bool (True/False)",
            "weight_ok": "WEIGHT_NIGHT_TOLERANCE (default: +5)",
            "weight_nok": "WEIGHT_NIGHT_NO_TOLERANCE (default: -20)",
        },
        "solver_handler": "soft terms in solver.py",
        "unit_tested": True,
    },

    "holiday_tolerance": {
        "label": "S4 – Tolleranza ai turni festivi",
        "description": (
            "Lavoratori con holiday_tolerance=True ricevono un bonus per turni in "
            "giorni festivi (weekend + festivi pubblici). Quelli con False ricevono penalità."
        ),
        "parameters": {
            "holiday_tolerance": "bool (True/False)",
            "weight_ok": "WEIGHT_HOLIDAY_TOLERANCE (default: +5)",
            "weight_nok": "WEIGHT_HOLIDAY_NO_TOLERANCE (default: -10)",
        },
        "solver_handler": "soft terms in solver.py",
        "unit_tested": True,
    },

    "preferred_rest_day": {
        "label": "S5 – Giorno di riposo preferito",
        "description": (
            "Un lavoratore può dichiarare un giorno della settimana preferito come "
            "giorno di riposo. Il sistema cerca di non assegnarlo in quel giorno "
            "(preferenza soft, non hard)."
        ),
        "parameters": {
            "preferred_rest_day": "intero 0-6 o null",
            "weight_met": "WEIGHT_REST_DAY_MET (default: +8)",
            "weight_missed": "-5 per ogni giorno di riposo perso",
        },
        "solver_handler": "soft terms in solver.py",
        "unit_tested": True,
    },
}


# ── Extension Pattern Registry ────────────────────────────────────────────────
# Vincoli NON presenti nel solver statico, ma implementati come blocchi testati
# in solver_extensions.py. Disponibili per la Fase 1b.

EXTENSION_PATTERNS: dict[str, dict[str, Any]] = {

    "max_consecutive_nights": {
        "label": "E1 – Massimo N notti consecutive",
        "description": (
            "Un lavoratore non può essere assegnato a più di N turni notturni "
            "consecutivi. Da usare quando H4 (2 giorni liberi dopo notte) non è "
            "sufficiente e si vuole un limite assoluto sulla sequenza di notti."
        ),
        "parameters": {
            "max_consecutive": "intero 1-7",
        },
        "solver_function": "add_max_consecutive_nights",
        "unit_tested": True,
        "risk_level": "LOW",
    },

    "min_gap_hours_between_shifts": {
        "label": "E2 – Pausa minima tra turni in ore",
        "description": (
            "Garantisce almeno N ore di pausa tra la fine di un turno e l'inizio "
            "del successivo per lo stesso lavoratore. Utile per norme contrattuali "
            "specifiche (es. 11 ore di riposo obbligatorio)."
        ),
        "parameters": {
            "min_gap_hours": "intero 8-24",
        },
        "solver_function": "add_min_gap_hours_between_shifts",
        "unit_tested": True,
        "risk_level": "LOW",
    },

    "pair_always_same_shift": {
        "label": "E3 – Due lavoratori sempre sullo stesso turno",
        "description": (
            "Garantisce che due lavoratori specifici siano sempre assegnati allo "
            "stesso turno (stesso giorno, stesso slot). Utile per coppie "
            "tutor-studente o coppie con complementarità clinica."
        ),
        "parameters": {
            "worker_a_id": "stringa worker_id",
            "worker_b_id": "stringa worker_id",
        },
        "solver_function": "add_pair_always_same_shift",
        "unit_tested": True,
        "risk_level": "MEDIUM",
    },

    "max_shifts_of_type_per_week": {
        "label": "E4 – Massimo N turni di un tipo per settimana",
        "description": (
            "Limita il numero di turni di un certo tipo (morning/afternoon/night) "
            "che un lavoratore può fare in una singola settimana."
        ),
        "parameters": {
            "shift_type": "'morning' | 'afternoon' | 'night'",
            "max_per_week": "intero 0-7",
            "applies_to": "'all' | lista di worker_id",
        },
        "solver_function": "add_max_shifts_of_type_per_week",
        "unit_tested": True,
        "risk_level": "LOW",
    },

    "worker_group_min_rest_days": {
        "label": "E5 – Giorni di riposo minimi per gruppo di lavoratori",
        "description": (
            "Garantisce che un gruppo di lavoratori abbia almeno N giorni liberi "
            "in un certo intervallo. Utile per gestire ferie collettive o chiusure "
            "reparti specifici."
        ),
        "parameters": {
            "worker_ids": "lista di worker_id",
            "min_rest_days": "intero 1-7",
            "in_day_range": "opzionale: [day_start, day_end]",
        },
        "solver_function": "add_worker_group_min_rest_days",
        "unit_tested": True,
        "risk_level": "MEDIUM",
    },
}


# ── Utility ───────────────────────────────────────────────────────────────────

def get_known_constraint_names() -> list[str]:
    """Restituisce i nomi chiave di tutti i vincoli noti."""
    return list(KNOWN_CONSTRAINTS.keys())


def get_extension_pattern_names() -> list[str]:
    """Restituisce i nomi chiave di tutti gli extension pattern disponibili."""
    return list(EXTENSION_PATTERNS.keys())


def build_classifier_context() -> str:
    """
    Costruisce il testo descrittivo dei Casi Noti e degli Extension Pattern
    da iniettare nel system prompt del Constraint Classifier Agent (Fase 1a/1b).
    """
    lines = ["=== VINCOLI NOTI (Casi Noti – Solver Statico) ===\n"]
    for key, v in KNOWN_CONSTRAINTS.items():
        lines.append(f"• [{key}] {v['label']}")
        lines.append(f"  {v['description']}")
        if v["parameters"]:
            params = ", ".join(f"{k}: {val}" for k, val in v["parameters"].items()
                               if not isinstance(val, dict))
            lines.append(f"  Parametri: {params}")
        lines.append("")

    lines.append("\n=== EXTENSION PATTERN (Fase 1b – Solo con avviso) ===\n")
    for key, v in EXTENSION_PATTERNS.items():
        lines.append(f"• [{key}] {v['label']} [Rischio: {v['risk_level']}]")
        lines.append(f"  {v['description']}")
        params = ", ".join(f"{k}: {val}" for k, val in v["parameters"].items())
        lines.append(f"  Parametri: {params}")
        lines.append("")

    return "\n".join(lines)
