#!/usr/bin/env python3
"""Train a tabular Q-learning agent across all 8 restaurant scenarios."""

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
    parser = argparse.ArgumentParser(description="Train tabular Q-learning agent")
    parser.add_argument("--episodes", type=int, default=600, help="Training episodes")
    parser.add_argument("--alpha", type=float, default=0.18, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.25, help="Initial epsilon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "q_table.json",
        help="Path to save learned Q-table",
    )
    parser.add_argument("--verbose", action="store_true", help="Print training progress")
    args = parser.parse_args()

    agent = TabularQLearningAgent(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
    )

    print(f"Training Q-learning agent for {args.episodes} episodes across {len(TASK_SPECS)} tasks...")
    metrics = agent.train(episodes=args.episodes, verbose=args.verbose)
    print(
        f"Training complete | mean_reward={metrics['mean_reward']:+.3f} | "
        f"mean_score={metrics['mean_score']:.2f} | states={int(metrics['states_learned'])}"
    )

    eval_report = agent.evaluate(verbose=False)
    average = eval_report["average"]["final_score"]
    print(f"Greedy evaluation (epsilon=0) average score: {average:.2f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.output)
    print(f"Saved Q-table to {args.output}")


if __name__ == "__main__":
    main()
