#!/usr/bin/env python3
"""Deterministic evaluation of a trained tabular Q-learning agent (epsilon=0)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.q_learning import TabularQLearningAgent  # noqa: E402
from env.tasks import TASK_SPECS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tabular Q-learning agent")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts" / "q_table.json",
        help="Path to saved Q-table JSON",
    )
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step episodes")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"Model not found at {args.model}. Run scripts/train_q_learning.py first."
        )

    agent = TabularQLearningAgent.load(args.model)
    agent.epsilon = 0.0

    print("Deterministic Q-learning evaluation (epsilon=0, greedy policy)\n")
    reports = agent.evaluate(task_ids=list(TASK_SPECS.keys()), verbose=args.verbose)

    for task_id in TASK_SPECS:
        report = reports[task_id]
        print(
            f"  {task_id:30s} score={report['final_score']:6.2f} | "
            f"profit={report['profit']:5.1f} rating={report['rating']:5.1f} "
            f"service={report['service']:5.1f}"
        )

    print(f"\n  {'average':30s} score={reports['average']['final_score']:6.2f}")


if __name__ == "__main__":
    main()
