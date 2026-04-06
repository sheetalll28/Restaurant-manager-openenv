"""
Deterministic graders for each task.

Each grader takes a ShiftResult and returns a score from 0 to 100.
No LLM is used — scoring is purely formulaic.

Grading pillars:
  1. Profit     — did the restaurant make money?
  2. Rating     — are customers happy? (1-5 stars)
  3. Service    — what fraction of orders were served?
  4. Satisfaction — the combined satisfaction score

Each task weights these pillars differently based on what matters most
for that scenario.
"""

from __future__ import annotations

from env.models import ShiftResult


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamp a value to [low, high] range."""
    return max(low, min(high, value))


def _profit_score(profit: float, target: float) -> float:
    """
    Score profit on a 0-100 scale.

    - profit >= target      → 100
    - profit == 0           → 50  (broke even)
    - profit <= -target     → 0   (lost as much as the target)
    """
    if target <= 0:
        return 50.0
    if profit >= target:
        return 100.0
    elif profit >= 0:
        # 0 to target → 50 to 100
        return 50.0 + (profit / target) * 50.0
    else:
        # -target to 0 → 0 to 50
        return _clamp(50.0 + (profit / target) * 50.0)


def _rating_score(rating: float, target: float) -> float:
    """
    Score rating on a 0-100 scale.

    - rating >= target  → 100
    - rating == 3.0     → 50  (mediocre)
    - rating == 1.0     → 0   (terrible)
    """
    if rating >= target:
        return 100.0
    elif rating >= 3.0:
        # 3.0 to target → 50 to 100
        range_size = target - 3.0
        if range_size <= 0:
            return 50.0
        return 50.0 + ((rating - 3.0) / range_size) * 50.0
    else:
        # 1.0 to 3.0 → 0 to 50
        return _clamp(((rating - 1.0) / 2.0) * 50.0)


def _service_score(served: int, failed: int) -> float:
    """
    Score service rate on a 0-100 scale.

    - 100% served → 100
    - 50% served  → 50
    - 0% served   → 0
    """
    total = served + failed
    if total == 0:
        return 50.0  # no customers = neutral
    return _clamp((served / total) * 100.0)


def _satisfaction_score(satisfaction: float) -> float:
    """Satisfaction is already 0-100, just clamp it."""
    return _clamp(satisfaction)


# ═══════════════════════════════════════════════════════════════════════
# TASK-SPECIFIC GRADERS
# ═══════════════════════════════════════════════════════════════════════


def grade_weekday_lunch(result: ShiftResult) -> dict:
    """
    Grade the easy task: Weekday Lunch Service.

    Targets (what a good agent should achieve):
      - Profit >= ₹8,000
      - Rating >= 4.2
      - Service rate >= 80%

    Weights: profit=30%, rating=25%, service=25%, satisfaction=20%
    """
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
    """
    Grade the medium task: Weekend Festival Rush.

    Targets:
      - Profit >= ₹12,000  (more customers = more potential revenue)
      - Rating >= 4.0      (must improve from starting 3.5)
      - Service rate >= 75% (harder with surge demand)

    Weights: profit=25%, rating=35%, service=25%, satisfaction=15%
    Rating is weighted highest because improving it is the key challenge.
    """
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
    """
    Grade the hard task: Crisis Management Shift.

    Targets:
      - Profit >= ₹5,000   (hard with doubled costs)
      - Rating >= 4.0      (must maintain despite pressure)
      - Service rate >= 70% (low stock makes this hard)

    Weights: profit=35%, rating=25%, service=15%, satisfaction=25%
    Profit is weighted highest because controlling costs is the key challenge.
    """
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


# ═══════════════════════════════════════════════════════════════════════
# REGISTRY — maps task_id to grader function
# ═══════════════════════════════════════════════════════════════════════

GRADER_REGISTRY: dict[str, callable] = {
    "weekday_lunch": grade_weekday_lunch,
    "weekend_rush": grade_weekend_rush,
    "crisis_shift": grade_crisis_shift,
}


def grade(task_id: str, result: ShiftResult) -> dict:
    """
    Grade a shift result for any task.

    Args:
        task_id: one of "weekday_lunch", "weekend_rush", "crisis_shift"
        result: the ShiftResult from env.get_result()

    Returns:
        dict with final_score (0-100), pillar_scores, weights, and raw result.
    """
    if task_id not in GRADER_REGISTRY:
        raise KeyError(
            f"No grader for task '{task_id}'. Available: {list(GRADER_REGISTRY.keys())}"
        )
    return GRADER_REGISTRY[task_id](result)
