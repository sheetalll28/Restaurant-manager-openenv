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
    env = RestaurantEnv()
    state = env.reset(task_id)
    done = False
    total_reward = 0.0
    while not done:
        action = policy(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if verbose:
            print(
                f"  Step {info['step']:2d} | {state.time_of_day} | "
                f"served={info['orders_served']} failed={info['orders_failed']} | "
                f"profit=₹{info['step_profit']:+.0f} | rating={state.customer_rating:.2f} | reward={reward:+.3f}"
            )
    result = env.get_result()
    grade_report = grade(task_id, result)
    if verbose:
        print(f"\n  FINAL SCORE: {grade_report['final_score']}/100")
    return result, grade_report