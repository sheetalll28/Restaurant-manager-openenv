"""
Simulation runner — executes a complete shift episode.

This is the main entry point for running the environment:
  result, grade_report = run_episode("weekday_lunch", policy_fn)
"""

from __future__ import annotations

from typing import Callable

from env.environment import RestaurantEnv
from env.graders import grade
from env.models import AgentAction, RestaurantState, ShiftResult


def run_episode(
    task_id: str,
    policy: Callable[[RestaurantState], AgentAction],
    verbose: bool = False,
) -> tuple[ShiftResult, dict]:
    """
    Run a complete shift from start to finish.

    Args:
        task_id: which task to run ("weekday_lunch", "weekend_rush", "crisis_shift")
        policy: a function that takes RestaurantState and returns AgentAction
        verbose: if True, print step-by-step info

    Returns:
        (result, grade_report) where:
          - result: ShiftResult with final metrics
          - grade_report: dict with final_score and pillar breakdowns
    """
    env = RestaurantEnv()
    state = env.reset(task_id)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Task: {task_id}")
        print(f"  Shift start: {state.time_of_day}")
        active = [s.name for s in state.staff if s.is_active]
        print(f"  Active staff: {', '.join(active)}")
        print(f"  Rating: {state.customer_rating} | Demand: {state.demand_level}x")
        print(f"{'='*60}\n")

    done = False
    total_reward = 0.0

    while not done:
        action = policy(state)
        state, reward, done, info = env.step(action)
        total_reward += reward

        if verbose:
            print(
                f"  Step {info['step']:2d} | {state.time_of_day} | "
                f"customers={info['customers_arrived']:2d} | "
                f"served={info['orders_served']:2d} failed={info['orders_failed']:2d} | "
                f"profit={info['step_profit']:+8.0f} | "
                f"rating={state.customer_rating:.2f} | "
                f"reward={reward:+.3f}"
            )
            if info.get("events"):
                for event in info["events"]:
                    print(f"         ↳ {event}")

    result = env.get_result()
    grade_report = grade(task_id, result)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  SHIFT COMPLETE")
        print(f"  Revenue:      {result.total_revenue:,.0f}")
        print(f"  Costs:        {result.total_costs:,.0f}")
        print(f"  Profit:       {result.profit:,.0f}")
        print(f"  Rating:       {result.average_rating:.2f}")
        print(f"  Served/Failed:{result.orders_served}/{result.orders_failed}")
        print(f"  Satisfaction: {result.customer_satisfaction:.1f}/100")
        print(f"  Total Reward: {total_reward:.3f}")
        print(f"  FINAL SCORE:  {grade_report['final_score']}/100")
        print(f"{'─'*60}\n")

    return result, grade_report
