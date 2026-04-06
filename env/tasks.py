"""
Task definitions for the restaurant management environment.

Each task function returns a dict of configuration that the Environment
will use to set up the initial state and simulation parameters.

Tasks represent increasing difficulty:
  1. weekday_lunch  — Easy: smooth, stable shift
  2. weekend_rush   — Medium: demand spike + low rating
  3. crisis_shift   — Hard: cost spike + competitor + inspection
"""

from __future__ import annotations

from env.models import InventoryItem, MenuItem, StaffMember


# ── Shared menu items (reused across tasks) ────────────────────────────

def _base_menu() -> list[MenuItem]:
    """Standard restaurant menu used as a starting point for all tasks."""
    return [
        MenuItem(
            name="Butter Chicken",
            price=350,
            cost=120,
            prep_time_minutes=20,
            category="main",
            ingredients={"chicken": 0.5, "spices": 0.1, "butter": 0.1},
        ),
        MenuItem(
            name="Paneer Tikka",
            price=280,
            cost=90,
            prep_time_minutes=15,
            category="main",
            ingredients={"paneer": 0.4, "spices": 0.1},
        ),
        MenuItem(
            name="Dal Tadka",
            price=180,
            cost=40,
            prep_time_minutes=12,
            category="main",
            ingredients={"lentils": 0.3, "spices": 0.05, "butter": 0.05},
        ),
        MenuItem(
            name="Naan",
            price=60,
            cost=15,
            prep_time_minutes=5,
            category="appetizer",
            ingredients={"flour": 0.2, "butter": 0.02},
        ),
        MenuItem(
            name="Gulab Jamun",
            price=120,
            cost=35,
            prep_time_minutes=8,
            category="dessert",
            ingredients={"flour": 0.1, "sugar": 0.15},
        ),
        MenuItem(
            name="Mango Lassi",
            price=100,
            cost=25,
            prep_time_minutes=3,
            category="drink",
            ingredients={"yogurt": 0.2, "sugar": 0.05},
        ),
    ]


# ── Shared staff pool ─────────────────────────────────────────────────

def _base_staff() -> list[StaffMember]:
    """Full staff roster. Tasks decide who starts active."""
    return [
        StaffMember(name="Ravi",   role="chef",       skill_level=0.9, hourly_wage=250),
        StaffMember(name="Priya",  role="chef",       skill_level=0.7, hourly_wage=200),
        StaffMember(name="Amit",   role="chef",       skill_level=0.5, hourly_wage=150),
        StaffMember(name="Sneha",  role="server",     skill_level=0.85, hourly_wage=150),
        StaffMember(name="Vikram", role="server",     skill_level=0.6, hourly_wage=120),
        StaffMember(name="Meera",  role="server",     skill_level=0.7, hourly_wage=130),
        StaffMember(name="Arjun",  role="dishwasher", skill_level=0.8, hourly_wage=100),
        StaffMember(name="Kavita", role="dishwasher", skill_level=0.5, hourly_wage=90),
    ]


# ── Shared inventory ──────────────────────────────────────────────────

def _base_inventory(cost_multiplier: float = 1.0) -> list[InventoryItem]:
    """
    Starting ingredient stock.

    Args:
        cost_multiplier: multiplies costs (1.0 = normal, 2.0 = doubled).
    """
    items = [
        InventoryItem(name="chicken",  quantity=30, unit="kg",     cost_per_unit=200,  reorder_cost_per_unit=300),
        InventoryItem(name="paneer",   quantity=20, unit="kg",     cost_per_unit=180,  reorder_cost_per_unit=270),
        InventoryItem(name="lentils",  quantity=25, unit="kg",     cost_per_unit=80,   reorder_cost_per_unit=120),
        InventoryItem(name="flour",    quantity=40, unit="kg",     cost_per_unit=40,   reorder_cost_per_unit=60),
        InventoryItem(name="butter",   quantity=15, unit="kg",     cost_per_unit=150,  reorder_cost_per_unit=225),
        InventoryItem(name="spices",   quantity=10, unit="kg",     cost_per_unit=300,  reorder_cost_per_unit=450),
        InventoryItem(name="sugar",    quantity=15, unit="kg",     cost_per_unit=50,   reorder_cost_per_unit=75),
        InventoryItem(name="yogurt",   quantity=20, unit="liters", cost_per_unit=60,   reorder_cost_per_unit=90),
    ]
    # Apply cost multiplier (for hard mode — ingredient price spike)
    if cost_multiplier != 1.0:
        items = [
            item.model_copy(update={
                "cost_per_unit": round(item.cost_per_unit * cost_multiplier, 2),
                "reorder_cost_per_unit": round(item.reorder_cost_per_unit * cost_multiplier, 2),
            })
            for item in items
        ]
    return items


# ═══════════════════════════════════════════════════════════════════════
# TASK DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════


def task_weekday_lunch() -> dict:
    """
    EASY — Stable weekday lunch service.

    - Normal demand (1.0×)
    - Full inventory
    - Good starting rating (4.3)
    - All key staff available
    - 12 steps (6-hour shift, 30 min each)

    The agent just needs to keep operations smooth.
    """
    staff = _base_staff()
    # Activate 2 chefs, 2 servers, 1 dishwasher for a normal lunch
    for s in staff:
        if s.name in ("Ravi", "Priya", "Sneha", "Vikram", "Arjun"):
            s.is_active = True

    return {
        "task_id": "weekday_lunch",
        "task_name": "Weekday Lunch Service",
        "difficulty": "easy",
        "total_steps": 12,
        "shift_start": "11:00",
        "step_duration_minutes": 30,
        "menu": _base_menu(),
        "staff": staff,
        "inventory": _base_inventory(),
        "initial_rating": 4.3,
        "demand_pattern": [1.0, 1.0, 1.2, 1.5, 1.5, 1.3, 1.0, 0.8, 0.7, 0.5, 0.3, 0.2],
        # ↑ demand peaks at lunch (steps 3-5), then fades toward evening
        "special_events": {},
    }


def task_weekend_rush() -> dict:
    """
    MEDIUM — Weekend festival rush with rating problems.

    - High demand (up to 2.5×)
    - Starting rating already low (3.5) — must improve it
    - One chef unavailable (Ravi is off)
    - 12 steps
    - Random large party at step 4

    The agent must handle surge demand with fewer resources
    and also improve customer satisfaction.
    """
    staff = _base_staff()
    # Ravi (best chef) is unavailable; activate others
    for s in staff:
        if s.name in ("Priya", "Amit", "Sneha", "Vikram", "Meera", "Arjun", "Kavita"):
            s.is_active = True

    return {
        "task_id": "weekend_rush",
        "task_name": "Weekend Festival Rush",
        "difficulty": "medium",
        "total_steps": 12,
        "shift_start": "11:00",
        "step_duration_minutes": 30,
        "menu": _base_menu(),
        "staff": staff,
        "inventory": _base_inventory(),
        "initial_rating": 3.5,  # already low — agent needs to fix this
        "demand_pattern": [1.5, 1.8, 2.0, 2.5, 2.5, 2.2, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8],
        # ↑ sustained high demand throughout the shift
        "special_events": {
            4: "large_party",  # at step 4, a big party of 15 arrives
        },
    }


def task_crisis_shift() -> dict:
    """
    HARD — Cost spike + competitor pressure + health inspection.

    - Ingredient costs doubled (cost_multiplier=2.0)
    - Competitor opened nearby: demand is lower and price-sensitive
    - Health inspection at step 8: must have clean kitchen + quality food
    - Starting rating decent (4.0) but fragile
    - Low starting inventory (agent must reorder at inflated prices)
    - 12 steps

    The agent faces financial pressure, competition, and regulatory risk
    all at once.
    """
    staff = _base_staff()
    # Only 1 chef and 1 server start active — agent must decide who to call in
    for s in staff:
        if s.name in ("Ravi", "Sneha", "Arjun"):
            s.is_active = True

    # Lower starting inventory to create scarcity
    inventory = _base_inventory(cost_multiplier=2.0)
    for item in inventory:
        item.quantity = round(item.quantity * 0.4, 1)  # only 40% of normal stock

    return {
        "task_id": "crisis_shift",
        "task_name": "Crisis Management Shift",
        "difficulty": "hard",
        "total_steps": 12,
        "shift_start": "11:00",
        "step_duration_minutes": 30,
        "menu": _base_menu(),
        "staff": staff,
        "inventory": inventory,
        "initial_rating": 4.0,
        "demand_pattern": [0.8, 0.9, 1.0, 1.2, 1.3, 1.2, 1.0, 0.9, 0.8, 0.7, 0.5, 0.3],
        # ↑ lower demand because of competitor, but still need to be profitable
        "special_events": {
            8: "health_inspection",  # at step 8, inspector arrives
        },
    }


# ── Registry: maps task_id -> task function ────────────────────────────

TASK_REGISTRY: dict[str, callable] = {
    "weekday_lunch": task_weekday_lunch,
    "weekend_rush": task_weekend_rush,
    "crisis_shift": task_crisis_shift,
}


def get_task(task_id: str) -> dict:
    """
    Look up a task by its ID and return its configuration dict.

    Raises KeyError if the task_id is not found.
    """
    if task_id not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task '{task_id}'. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]()
