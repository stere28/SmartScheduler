"""
SmartScheduler – Main Entry Point
====================================
Runs the full multi-agent LangGraph pipeline.

Usage:
    python main.py --use-case A             # Use Case A
    python main.py --use-case B             # Use Case B
    python main.py --use-case A --no-llm    # Skip LLM calls (solver-only mode)
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _check_provider() -> str:
    """Return the active LLM provider name from env."""
    return os.environ.get("LLM_PROVIDER", "ollama").lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartScheduler – Multi-Agent Scheduler")
    parser.add_argument(
        "--use-case", choices=["A", "B"], default="A",
        help="Use Case A (homogeneous, 10 workers) or B (std+specialized, 16 workers)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM agents; run the CP-SAT solver directly (no provider required).",
    )
    parser.add_argument(
        "--time-limit", type=int, default=120,
        help="CP-SAT solver time limit in seconds (default: 120)",
    )
    args = parser.parse_args()

    provider = _check_provider()
    print(f"\n  [Config] LLM Provider : {provider}")
    print(f"  [Config] Model        : {os.environ.get('LLM_MODEL', 'llama3.2')}")

    if args.no_llm:
        print("\n  [!] --no-llm flag set: skipping LLM pipeline.")
        return

    # Full LangGraph pipeline
    print(f"\n  Starting SmartScheduler pipeline (Use Case {args.use_case})…\n")
    from pipeline import run_pipeline
    from output import print_schedule, print_worker_stats, export_csv, export_json

    final_state = run_pipeline(use_case=args.use_case)

    print(f"\n{'='*70}")
    print("  Pipeline complete.")
    print(f"{'='*70}")

    print_worker_stats(final_state)
    print_schedule(final_state)

    suffix = args.use_case
    export_csv(final_state, path=f"schedule_uc{suffix}.csv")
    export_json(final_state, path=f"schedule_uc{suffix}.json")

    print("\n  Pipeline log:")
    for entry in final_state.history:
        print(f"    {entry}")


if __name__ == "__main__":
    main()
