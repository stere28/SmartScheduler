# SmartScheduler

A multi-agent system for fair, constraint-aware hospital shift scheduling.  
Combines **LangChain / LangGraph** LLM agents with **Google OR-Tools CP-SAT** symbolic optimization.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LangGraph Pipeline                              │
│                                                                     │
│  ┌──────────────┐   ┌─────────────┐   ┌──────────────────┐         │
│  │  Stage 1     │──▶│  Stage 2    │──▶│    Stage 3       │         │
│  │  Preferences │   │  Drafting   │   │  Verification    │         │
│  │  Agent (LLM) │   │  Agent      │   │  Agent           │         │
│  │              │   │  (LLM+CPSAT)│   │  (Symbolic)      │         │
│  └──────────────┘   └─────────────┘   └────────┬─────────┘         │
│                                                 │                   │
│                                        pass?    │    fail?          │
│                                           ▼           ▼            │
│                                   ┌──────────────┐  END            │
│                                   │   Stage 4    │                 │
│                                   │  Refinement  │◀──┐             │
│                                   │  Agent       │   │             │
│                                   └──────┬───────┘   │             │
│                                          │ re-verify  │             │
│                                          └────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

### Stage 1 – Preferences Agent
- Workers describe preferences in natural language
- LLM extracts structured `ShiftPreference` objects (preferred/avoided shifts, tolerances, availability, rest-day wishes)

### Stage 2 – Drafting Agent
- LLM formulates scheduling strategy
- **CP-SAT solver** generates the initial schedule satisfying all hard constraints

### Stage 3 – Verification Agent
- Symbolic checker validates every hard constraint
- Computes per-worker fairness/satisfaction scores (0–100)
- Identifies the most disadvantaged worker

### Stage 4 – Refinement Agent
- LLM proposes targeted improvements for the worst-off worker
- CP-SAT re-solves with a satisfaction floor protecting other workers
- Loop repeats until no improvement is possible or max iterations reached

---

## Hard Constraints (CP-SAT)

| ID | Constraint |
|----|-----------|
| H1 | Minimum staffing per shift (Use Case A: ≥2; Use Case B: ≥1 spec, ≥1 std, ≥3 total) |
| H2 | At most 1 shift per worker per day |
| H3 | No consecutive cross-day shifts (night(d) → morning(d+1) forbidden) |
| H4 | 2 mandatory free days after each night shift |
| H5 | Exactly 25 workload units per worker per month (night = 2 units) |
| H6 | Maximum 36 working hours per week |
| H7 | Worker unavailability from preferences |

## Soft Preferences (Objective Function)

| Preference | Points |
|-----------|--------|
| Assigned preferred shift | +10 |
| Assigned avoided shift | −15 |
| Night shift with tolerance | +5 |
| Night shift without tolerance | −20 |
| Holiday shift with tolerance | +5 |
| Holiday shift without tolerance | −10 |
| Preferred rest day granted | +8 |
| Working on preferred rest day | −5 |

**Objective**: Maximize `0.7 × Σ satisfaction + 0.3 × n_workers × min_satisfaction` (max-min fairness)

---

## Scheduling Horizon

**7 December 2026 → 6 January 2027** (31 days, 93 shifts)

Public holidays treated specially: 25 Dec, 26 Dec, 1 Jan + all weekends.

---

## Use Cases

### Use Case A – Homogeneous Workers
- 10 standard workers
- Any worker can cover any shift
- Minimum **2 workers** per shift

### Use Case B – Standard + Specialized Workers
- 10 standard + 6 specialized workers
- Minimum per shift: **≥1 specialized**, **≥1 standard**, **≥3 total**
- Specialized workers can substitute in the standard role

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Mode 1: Standalone Solver (no API key needed)
```bash
# Use Case A
python solve_only.py --use-case A

# Use Case B
python solve_only.py --use-case B --time-limit 180
```

### Mode 2: Full LangGraph Pipeline (requires OpenAI API key)
```bash
# Copy and fill in the API key
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-...

# Run Use Case A
python main.py --use-case A

# Run Use Case B
python main.py --use-case B
```

### Mode 3: Auto-detect (main.py without LLM → solver-only fallback)
```bash
python main.py --use-case A --no-llm
```

---

## Output Files

| File | Description |
|------|-------------|
| `schedule_ucA.csv` | Use Case A schedule in CSV format |
| `schedule_ucA.json` | Use Case A full state (schedule + fairness + log) |
| `schedule_ucB.csv` | Use Case B schedule in CSV format |
| `schedule_ucB.json` | Use Case B full state |

---

## Project Structure

```
SmartScheduler/
├── config.py          # Global constants, horizon, constraint parameters
├── models.py          # Pydantic data models (preferences, schedule, state)
├── solver.py          # OR-Tools CP-SAT scheduling model
├── agents.py          # Four LangChain agents (Stage 1-4)
├── pipeline.py        # LangGraph pipeline wiring agents
├── output.py          # Pretty-printer, CSV/JSON exporter
├── solve_only.py      # Standalone solver entry point (no LLM)
├── main.py            # Main entry point (full pipeline or solver-only)
├── requirements.txt
└── .env.example
```

---

## Sample Output (Use Case A)

```
Worker Statistics

Worker  Type      Mrn  Aft  Ngt  Workload  Hours  NightHol
------  --------  ---  ---  ---  --------  -----  --------
W01     standard    9    0    2        13     66         1
W02     standard    2    6    3        17     66         2
...

Fairness / Satisfaction Scores

Worker  Score (0-100)  Bar
------  -------------  --------------------
W09             41.2   ████████░░░░░░░░░░░░
W04             42.5   ████████░░░░░░░░░░░░
...
W06            100.0   ████████████████████
```
