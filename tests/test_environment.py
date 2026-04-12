from env.environment import RestaurantEnv
from env.models import AgentAction
from env.tasks import TASK_SPECS, list_task_metadata


def test_all_registered_tasks_reset():
    env = RestaurantEnv()
    for task_id in TASK_SPECS:
        state = env.reset(task_id)
        assert state.total_steps == TASK_SPECS[task_id]["total_steps"]
        assert state.step == 0


def test_step_returns_reward_breakdown():
    env = RestaurantEnv()
    env.reset("weekday_lunch")
    observation, reward, done, info = env.step(AgentAction())
    assert isinstance(reward, float)
    assert done is False
    assert "reward_breakdown" in info
    assert "penalty" in info["reward_breakdown"]
    assert observation.step == 1


def test_result_contains_operational_metrics():
    env = RestaurantEnv()
    env.reset("crisis_shift")
    for _ in range(3):
        env.step(AgentAction())
    result = env.get_result()
    assert result.labor_costs >= 0
    assert result.food_costs >= 0
    assert result.reorder_costs >= 0
    assert result.stockout_failures >= 0
    assert result.capacity_failures >= 0


def test_task_metadata_exposes_new_scenarios():
    task_ids = {task["id"] for task in list_task_metadata()}
    assert "monsoon_delivery_crunch" in task_ids
    assert "wedding_catering_chaos" in task_ids

