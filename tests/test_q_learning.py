"""Tests for tabular Q-learning agent."""

from __future__ import annotations

import random

from agents.discretization import ACTION_COUNT, decode_action, encode_state
from agents.q_learning import TabularQLearningAgent
from env.environment import RestaurantEnv
from env.models import AgentAction
from env.policies import do_nothing_policy, simple_rule_policy
from env.simulation import run_episode
from env.tasks import TASK_SPECS


def test_encode_state_is_hashable_and_stable():
    env = RestaurantEnv()
    state = env.reset("weekday_lunch")
    key = encode_state(state)
    assert isinstance(key, tuple)
    assert len(key) == 7
    assert encode_state(state) == key


def test_decode_action_returns_valid_agent_action():
    env = RestaurantEnv()
    state = env.reset("weekday_lunch")
    for action_index in range(ACTION_COUNT):
        action = decode_action(action_index, state)
        assert isinstance(action, AgentAction)


def test_td_update_moves_q_value_toward_target():
    agent = TabularQLearningAgent(alpha=0.5, gamma=0.9, epsilon=0.0, seed=1)
    env = RestaurantEnv()
    state = env.reset("weekday_lunch")
    action_index = 0
    next_state, reward, done, _info = env.step(decode_action(action_index, state))

    state_key = encode_state(state)
    before = agent._ensure_state(state_key)[action_index]
    td_error = agent.td_update(state, action_index, reward, next_state, done)
    after = agent.q_table[state_key][action_index]

    assert td_error != 0.0 or reward == 0.0
    assert after != before or reward == 0.0


def test_epsilon_greedy_explores_then_greedy():
    agent = TabularQLearningAgent(epsilon=1.0, seed=7)
    env = RestaurantEnv()
    state = env.reset("weekday_lunch")

    seen = {agent.select_action(state, explore=True) for _ in range(30)}
    assert len(seen) > 1

    state_key = encode_state(state)
    agent.q_table[state_key] = [0.0] * ACTION_COUNT
    agent.q_table[state_key][3] = 10.0
    agent.epsilon = 0.0
    assert agent.select_action(state, explore=False) == 3


def test_training_improves_over_random_baseline():
    random_agent = TabularQLearningAgent(epsilon=1.0, seed=99)
    random_policy = random_agent.as_policy(explore=True)
    random_scores = []
    for task_id in list(TASK_SPECS.keys())[:3]:
        _result, report = run_episode(task_id, random_policy)
        random_scores.append(report["final_score"])
    random_avg = sum(random_scores) / len(random_scores)

    learner = TabularQLearningAgent(alpha=0.2, epsilon=0.3, seed=42)
    learner.train(episodes=180, task_ids=["weekday_lunch", "weekend_rush", "crisis_shift"])
    learner.epsilon = 0.0
    learned_scores = []
    for task_id in ["weekday_lunch", "weekend_rush", "crisis_shift"]:
        _result, report = run_episode(task_id, learner.as_policy(explore=False))
        learned_scores.append(report["final_score"])
    learned_avg = sum(learned_scores) / len(learned_scores)

    assert learned_avg >= random_avg


def test_q_agent_beats_do_nothing_on_easy_task():
    agent = TabularQLearningAgent(alpha=0.2, epsilon=0.25, seed=11)
    agent.train(episodes=120, task_ids=["weekday_lunch"])
    agent.epsilon = 0.0

    _result, learned = run_episode("weekday_lunch", agent.as_policy(explore=False))
    _result, baseline = run_episode("weekday_lunch", do_nothing_policy)

    assert learned["final_score"] >= baseline["final_score"]


def test_save_and_load_roundtrip(tmp_path):
    agent = TabularQLearningAgent(seed=5)
    env = RestaurantEnv()
    state = env.reset("office_catering_lunch")
    action_index = agent.select_action(state, explore=True)
    next_state, reward, done, _info = env.step(decode_action(action_index, state))
    agent.td_update(state, action_index, reward, next_state, done)

    path = tmp_path / "q_table.json"
    agent.save(path)
    loaded = TabularQLearningAgent.load(path)

    assert loaded.q_table == agent.q_table
    assert loaded.alpha == agent.alpha
    assert loaded.gamma == agent.gamma
