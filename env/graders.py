from __future__ import annotations

from env.models import ShiftResult
from env.tasks import TASK_SPECS


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _profit_score(profit: float, target: float) -> float:
    if target <= 0:
        return 50.0
    if profit >= target:
        overshoot = min((profit - target) / max(target, 1.0), 0.4)
        return _clamp(100.0 - overshoot * 5.0)
    if profit >= 0:
        return 45.0 + (profit / target) * 55.0
    return _clamp(45.0 + (profit / target) * 70.0)


def _rating_score(rating: float, target: float) -> float:
    if rating >= target:
        return 100.0
    floor = 3.0
    if rating >= floor:
        span = max(target - floor, 0.25)
        return 40.0 + ((rating - floor) / span) * 60.0
    return _clamp(((rating - 1.0) / 2.0) * 40.0)


def _service_score(served: int, failed: int) -> float:
    total = served + failed
    if total == 0:
        return 30.0
    success_rate = served / total
    failure_rate = failed / total
    return _clamp(success_rate * 100.0 - (failure_rate**2) * 25.0)


def _satisfaction_score(satisfaction: float) -> float:
    return _clamp(satisfaction)


def _efficiency_score(result: ShiftResult) -> float:
    if result.total_revenue <= 0:
        return 0.0
    margin = result.profit / max(result.total_revenue, 1.0)
    stockout_penalty = min(result.stockout_failures * 3.0, 35.0)
    capacity_penalty = min(result.capacity_failures * 1.5, 25.0)
    reorder_penalty = min(result.reorder_costs / 300.0, 20.0)
    labor_penalty = min(result.labor_costs / 800.0, 20.0)
    base = _clamp(55.0 + margin * 60.0)
    return _clamp(base - stockout_penalty - capacity_penalty - reorder_penalty - labor_penalty)


def _weights_for_task(task_id: str) -> dict[str, float]:
    if task_id == "weekend_rush":
        return {
            "profit": 0.22,
            "rating": 0.28,
            "service": 0.22,
            "satisfaction": 0.13,
            "efficiency": 0.15,
        }
    if task_id == "crisis_shift":
        return {
            "profit": 0.28,
            "rating": 0.18,
            "service": 0.16,
            "satisfaction": 0.18,
            "efficiency": 0.20,
        }
    if task_id == "wedding_catering_chaos":
        return {
            "profit": 0.26,
            "rating": 0.20,
            "service": 0.24,
            "satisfaction": 0.10,
            "efficiency": 0.20,
        }
    return {
        "profit": 0.24,
        "rating": 0.22,
        "service": 0.22,
        "satisfaction": 0.14,
        "efficiency": 0.18,
    }


def grade(task_id: str, result: ShiftResult) -> dict:
    if task_id not in TASK_SPECS:
        raise KeyError(f"No grader for task '{task_id}'. Available: {list(TASK_SPECS.keys())}")

    targets = TASK_SPECS[task_id]["targets"]
    scores = {
        "profit": _profit_score(result.profit, target=targets["profit"]),
        "rating": _rating_score(result.average_rating, target=targets["rating"]),
        "service": _service_score(result.orders_served, result.orders_failed),
        "satisfaction": _satisfaction_score(result.customer_satisfaction),
        "efficiency": _efficiency_score(result),
    }
    weights = _weights_for_task(task_id)
    final = sum(scores[key] * weights[key] for key in scores)

    return {
        "task_id": task_id,
        "final_score": round(_clamp(final), 2),
        "pillar_scores": {key: round(value, 2) for key, value in scores.items()},
        "weights": weights,
        "result": result.model_dump(),
    }

