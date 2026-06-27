#!/usr/bin/env python3
"""Evaluate rule-based baseline policies across all 8 scenarios."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.policies import (  # noqa: E402
    do_nothing_policy,
    profit_first_policy,
    service_first_policy,
    simple_rule_policy,
)
from env.simulation import run_episode  # noqa: E402
from env.tasks import TASK_SPECS  # noqa: E402

POLICIES = {
    "do_nothing": do_nothing_policy,
    "simple_rule": simple_rule_policy,
    "profit_first": profit_first_policy,
    "service_first": service_first_policy,
}


def main() -> None:
    task_ids = list(TASK_SPECS.keys())
    summary: dict[str, float] = {}

    print("Baseline benchmark (deterministic evaluation)\n")
    for policy_name, policy_fn in POLICIES.items():
        scores: list[float] = []
        print(f"{policy_name}:")
        for task_id in task_ids:
            _result, grade_report = run_episode(task_id, policy_fn)
            score = grade_report["final_score"]
            scores.append(score)
            print(f"  {task_id:30s} {score:6.2f}")
        average = sum(scores) / len(scores)
        summary[policy_name] = round(average, 2)
        print(f"  {'average':30s} {average:6.2f}\n")

    print("Average score across all 8 tasks:")
    for policy_name, average in sorted(summary.items(), key=lambda item: item[1], reverse=True):
        print(f"  {policy_name:15s} {average:6.2f}")


if __name__ == "__main__":
    main()
