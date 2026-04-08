from __future__ import annotations
from env.models import ShiftResult

def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))

def _profit_score(profit: float, target: float) -> float:
    if target <= 0:
        return 50.0
    if profit >= target:
        return 100.0
    elif profit >= 0:
        return 50.0 + (profit / target) * 50.0
    else:
        return _clamp(50.0 + (profit / target) * 50.0)

def _rating_score(rating: float, target: float) -> float:
    if rating >= target:
        return 100.0
    elif rating >= 3.0:
        range_size = target - 3.0
        if range_size <= 0:
            return 50.0
        return 50.0 + ((rating - 3.0) / range_size) * 50.0
    else:
        return _clamp(((rating - 1.0) / 2.0) * 50.0)

def _service_score(served: int, failed: int) -> float:
    total = served + failed
    if total == 0:
        return 50.0
    return _clamp((served / total) * 100.0)

def _satisfaction_score(satisfaction: float) -> float:
    return _clamp(satisfaction)

def grade_weekday_lunch(result: ShiftResult) -> dict:
    scores = {
        "profit": _profit_score(result.profit, target=8000),
        "rating": _rating_score(result.average_rating, target=4.2),
        "service": _service_score(result.orders_served, result.orders_failed),
        "satisfaction": _satisfaction_score(result.customer_satisfaction),
    }
    weights = {"profit": 0.30, "rating": 0.25, "service": 0.25, "satisfaction": 0.20}
    final = sum(scores[k] * weights[k] for k in scores)
    return {
        "task_id": "weekday_lunch",
        "final_score": round(_clamp(final), 2),
        "pillar_scores": {k: round(v, 2) for k, v in scores.items()},
        "weights": weights,
        "result": result.model_dump(),
    }

def grade_weekend_rush(result: ShiftResult) -> dict:
    scores = {
        "profit": _profit_score(result.profit, target=12000),
        "rating": _rating_score(result.average_rating, target=4.0),
        "service": _service_score(result.orders_served, result.orders_failed),
        "satisfaction": _satisfaction_score(result.customer_satisfaction),
    }
    weights = {"profit": 0.25, "rating": 0.35, "service": 0.25, "satisfaction": 0.15}
    final = sum(scores[k] * weights[k] for k in scores)
    return {
        "task_id": "weekend_rush",
        "final_score": round(_clamp(final), 2),
        "pillar_scores": {k: round(v, 2) for k, v in scores.items()},
        "weights": weights,
        "result": result.model_dump(),
    }

def grade_crisis_shift(result: ShiftResult) -> dict:
    scores = {
        "profit": _profit_score(result.profit, target=5000),
        "rating": _rating_score(result.average_rating, target=4.0),
        "service": _service_score(result.orders_served, result.orders_failed),
        "satisfaction": _satisfaction_score(result.customer_satisfaction),
    }
    weights = {"profit": 0.35, "rating": 0.25, "service": 0.15, "satisfaction": 0.25}
    final = sum(scores[k] * weights[k] for k in scores)
    return {
        "task_id": "crisis_shift",
        "final_score": round(_clamp(final), 2),
        "pillar_scores": {k: round(v, 2) for k, v in scores.items()},
        "weights": weights,
        "result": result.model_dump(),
    }

GRADER_REGISTRY: dict = {
    "weekday_lunch": grade_weekday_lunch,
    "weekend_rush": grade_weekend_rush,
    "crisis_shift": grade_crisis_shift,
}

def grade(task_id: str, result: ShiftResult) -> dict:
    if task_id not in GRADER_REGISTRY:
        raise KeyError(f"No grader for task '{task_id}'. Available: {list(GRADER_REGISTRY.keys())}")
    return GRADER_REGISTRY[task_id](result)