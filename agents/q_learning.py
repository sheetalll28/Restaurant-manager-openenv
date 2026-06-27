"""
Tabular Q-learning agent with TD(0) updates and epsilon-greedy exploration.

Uses the standard off-policy Q-learning update:

    Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agents.discretization import ACTION_COUNT, decode_action, encode_state
from env.environment import RestaurantEnv
from env.graders import grade
from env.models import AgentAction, RestaurantState
from env.tasks import TASK_SPECS


@dataclass
class TabularQLearningAgent:
    """Tabular Q-learning agent over discretized restaurant states."""

    alpha: float = 0.15
    gamma: float = 0.95
    epsilon: float = 0.20
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.995
    seed: int = 42
    q_table: dict[tuple[int, ...], list[float]] = field(default_factory=dict)
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def _ensure_state(self, state_key: tuple[int, ...]) -> list[float]:
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * ACTION_COUNT
        return self.q_table[state_key]

    def select_action(self, state: RestaurantState, *, explore: bool | None = None) -> int:
        """Select an action with epsilon-greedy exploration."""
        state_key = encode_state(state)
        q_values = self._ensure_state(state_key)
        should_explore = self.epsilon > 0.0 if explore is None else explore

        if should_explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(ACTION_COUNT)

        best_value = max(q_values)
        best_actions = [index for index, value in enumerate(q_values) if value == best_value]
        return self.rng.choice(best_actions)

    def td_update(
        self,
        state: RestaurantState,
        action_index: int,
        reward: float,
        next_state: RestaurantState,
        done: bool,
    ) -> float:
        """Apply one Q-learning TD update and return the TD error."""
        state_key = encode_state(state)
        next_key = encode_state(next_state)
        q_values = self._ensure_state(state_key)
        next_q_values = self._ensure_state(next_key)

        current = q_values[action_index]
        bootstrap = 0.0 if done else max(next_q_values)
        target = reward + self.gamma * bootstrap
        td_error = target - current
        q_values[action_index] = current + self.alpha * td_error
        return td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def as_policy(self, *, explore: bool = False) -> Callable[[RestaurantState], AgentAction]:
        """Return a policy function compatible with env.simulation.run_episode."""

        def policy(state: RestaurantState) -> AgentAction:
            action_index = self.select_action(state, explore=explore)
            return decode_action(action_index, state)

        return policy

    def train_episode(self, env: RestaurantEnv, task_id: str) -> tuple[float, float]:
        """Run one training episode and update the Q-table online."""
        state = env.reset(task_id)
        total_reward = 0.0

        while True:
            action_index = self.select_action(state, explore=True)
            action = decode_action(action_index, state)
            next_state, reward, done, _info = env.step(action)
            self.td_update(state, action_index, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break

        result = env.get_result()
        grade_report = grade(task_id, result)
        return total_reward, grade_report["final_score"]

    def train(
        self,
        *,
        episodes: int = 400,
        task_ids: list[str] | None = None,
        verbose: bool = False,
    ) -> dict[str, float]:
        """Train across scenarios and return summary metrics."""
        tasks = task_ids or list(TASK_SPECS.keys())
        env = RestaurantEnv()
        episode_rewards: list[float] = []
        episode_scores: list[float] = []

        for episode in range(episodes):
            task_id = tasks[episode % len(tasks)]
            total_reward, final_score = self.train_episode(env, task_id)
            episode_rewards.append(total_reward)
            episode_scores.append(final_score)
            self.decay_epsilon()

            if verbose and (episode + 1) % max(episodes // 10, 1) == 0:
                window = episode_rewards[-max(len(tasks), 1) :]
                print(
                    f"  episode {episode + 1:4d}/{episodes} | "
                    f"avg_reward={sum(window) / len(window):+.3f} | "
                    f"epsilon={self.epsilon:.3f} | "
                    f"states={len(self.q_table)}"
                )

        return {
            "episodes": float(episodes),
            "mean_reward": sum(episode_rewards) / len(episode_rewards),
            "mean_score": sum(episode_scores) / len(episode_scores),
            "states_learned": float(len(self.q_table)),
            "final_epsilon": self.epsilon,
        }

    def evaluate(
        self,
        task_ids: list[str] | None = None,
        *,
        verbose: bool = False,
    ) -> dict[str, dict[str, float]]:
        """Deterministic greedy evaluation (epsilon = 0) across tasks."""
        from env.simulation import run_episode

        tasks = task_ids or list(TASK_SPECS.keys())
        policy = self.as_policy(explore=False)
        reports: dict[str, dict[str, float]] = {}

        for task_id in tasks:
            _result, grade_report = run_episode(task_id, policy, verbose=verbose)
            reports[task_id] = {
                "final_score": grade_report["final_score"],
                "profit": grade_report["pillar_scores"]["profit"],
                "rating": grade_report["pillar_scores"]["rating"],
                "service": grade_report["pillar_scores"]["service"],
            }

        reports["average"] = {
            "final_score": round(
                sum(report["final_score"] for key, report in reports.items() if key != "average")
                / len(tasks),
                2,
            )
        }
        return reports

    def save(self, path: str | Path) -> None:
        payload = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "seed": self.seed,
            "q_table": {json.dumps(list(key)): values for key, values in self.q_table.items()},
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TabularQLearningAgent:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
            seed=payload["seed"],
        )
        agent.q_table = {
            tuple(json.loads(key)): values for key, values in payload["q_table"].items()
        }
        return agent
