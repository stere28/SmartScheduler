"""
SmartScheduler - Global Configuration
======================================
Scheduling horizon: 7 Dec 2026 → 6 Jan 2027 (31 days)
"""

from datetime import date, timedelta

# ── Scheduling Horizon ─────────────────────────────────────────────────────────
START_DATE = date(2026, 12, 7)
END_DATE   = date(2027, 1, 6)
NUM_DAYS   = (END_DATE - START_DATE).days + 1   # 31

DATES = [START_DATE + timedelta(days=d) for d in range(NUM_DAYS)]

# ── Shifts ─────────────────────────────────────────────────────────────────────
# 0 = morning (08-14), 1 = afternoon (14-20), 2 = night (20-08)
SHIFT_NAMES   = ["morning", "afternoon", "night"]
SHIFT_HOURS   = [6, 6, 12]          # actual duration in hours
SHIFT_WEIGHT  = [1, 1, 2]           # workload units (night counts double)
NUM_SHIFTS    = 3

# ── Constraints ────────────────────────────────────────────────────────────────
MAX_HOURS_PER_WEEK    = 36
TARGET_SHIFTS_MONTH   = 25          # each worker must cover exactly 25 shifts
FREE_DAYS_AFTER_NIGHT = 2           # mandatory free days after a night shift
MAX_SHIFTS_PER_DAY    = 1
NO_CONSECUTIVE_SHIFTS = True        # cannot cover two subsequent shifts same day

# ── Use Case A (homogeneous) ───────────────────────────────────────────────────
UC_A = {
    "num_workers": 10,
    "worker_types": ["standard"] * 10,
    "min_standard_per_shift": 2,
    "min_specialized_per_shift": 0,
}

# ── Use Case B (standard + specialized) ───────────────────────────────────────
UC_B = {
    "num_standard_workers":     10,
    "num_specialized_workers":  6,
    "num_workers":              16,
    "worker_types": ["standard"] * 10 + ["specialized"] * 6,
    "min_standard_per_shift":   2,   # at least 2 std (specialized can substitute)
    "min_specialized_per_shift":1,
}

# ── LLM Settings ──────────────────────────────────────────────────────────────
# All values are read from environment variables at runtime (see llm_provider.py).
# These constants serve as fallback defaults only.
import os as _os

LLM_PROVIDER    = _os.environ.get("LLM_PROVIDER",    "ollama")           # "ollama" | "openai"
LLM_MODEL       = _os.environ.get("LLM_MODEL",       "llama3.2")         # model name
LLM_TEMPERATURE = float(_os.environ.get("LLM_TEMPERATURE", "0.2"))
OLLAMA_BASE_URL = _os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

MAX_REFINEMENT_ITERATIONS = 8

# ── Fairness ──────────────────────────────────────────────────────────────────
# Satisfaction score weights (soft preferences contribute to a 0-100 score)
WEIGHT_PREFERRED_SHIFT    = 10   # assigned a preferred shift
WEIGHT_AVOIDED_SHIFT      = -15  # assigned an avoided shift
WEIGHT_NIGHT_TOLERANCE    = 5    # worker declared night tolerance → bonus per night
WEIGHT_NIGHT_NO_TOLERANCE = -20  # worker declared no night tolerance → malus
WEIGHT_HOLIDAY_TOLERANCE  = 5
WEIGHT_HOLIDAY_NO_TOLERANCE = -10
WEIGHT_REST_DAY_MET       = 8    # preferred rest day granted

