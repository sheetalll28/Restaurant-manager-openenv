from __future__ import annotations

from typing import Any

from env.models import InventoryItem, MenuItem, StaffMember


def _base_menu() -> list[MenuItem]:
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


def _base_staff() -> list[StaffMember]:
    return [
        StaffMember(name="Ravi", role="chef", skill_level=0.9, hourly_wage=250),
        StaffMember(name="Priya", role="chef", skill_level=0.7, hourly_wage=200),
        StaffMember(name="Amit", role="chef", skill_level=0.5, hourly_wage=150),
        StaffMember(name="Sneha", role="server", skill_level=0.85, hourly_wage=150),
        StaffMember(name="Vikram", role="server", skill_level=0.6, hourly_wage=120),
        StaffMember(name="Meera", role="server", skill_level=0.7, hourly_wage=130),
        StaffMember(name="Arjun", role="dishwasher", skill_level=0.8, hourly_wage=100),
        StaffMember(name="Kavita", role="dishwasher", skill_level=0.5, hourly_wage=90),
    ]


def _base_inventory(
    *,
    cost_multiplier: float = 1.0,
    quantity_multiplier: float = 1.0,
) -> list[InventoryItem]:
    items = [
        InventoryItem(
            name="chicken",
            quantity=30,
            unit="kg",
            cost_per_unit=200,
            reorder_cost_per_unit=300,
        ),
        InventoryItem(
            name="paneer",
            quantity=20,
            unit="kg",
            cost_per_unit=180,
            reorder_cost_per_unit=270,
        ),
        InventoryItem(
            name="lentils",
            quantity=25,
            unit="kg",
            cost_per_unit=80,
            reorder_cost_per_unit=120,
        ),
        InventoryItem(
            name="flour",
            quantity=40,
            unit="kg",
            cost_per_unit=40,
            reorder_cost_per_unit=60,
        ),
        InventoryItem(
            name="butter",
            quantity=15,
            unit="kg",
            cost_per_unit=150,
            reorder_cost_per_unit=225,
        ),
        InventoryItem(
            name="spices",
            quantity=10,
            unit="kg",
            cost_per_unit=300,
            reorder_cost_per_unit=450,
        ),
        InventoryItem(
            name="sugar",
            quantity=15,
            unit="kg",
            cost_per_unit=50,
            reorder_cost_per_unit=75,
        ),
        InventoryItem(
            name="yogurt",
            quantity=20,
            unit="liters",
            cost_per_unit=60,
            reorder_cost_per_unit=90,
        ),
    ]
    if cost_multiplier != 1.0 or quantity_multiplier != 1.0:
        items = [
            item.model_copy(
                update={
                    "cost_per_unit": round(item.cost_per_unit * cost_multiplier, 2),
                    "reorder_cost_per_unit": round(
                        item.reorder_cost_per_unit * cost_multiplier, 2
                    ),
                    "quantity": round(item.quantity * quantity_multiplier, 2),
                }
            )
            for item in items
        ]
    return items


def _activate(staff: list[StaffMember], names: tuple[str, ...]) -> list[StaffMember]:
    active_names = set(names)
    for member in staff:
        member.is_active = member.name in active_names
    return staff


TASK_SPECS: dict[str, dict[str, Any]] = {
    "weekday_lunch": {
        "name": "Weekday Lunch Service",
        "difficulty": "easy",
        "description": (
            "Stable lunch service with a predictable peak, healthy reputation, "
            "and enough inventory to reward sensible staffing."
        ),
        "targets": {"profit": 8000, "rating": 4.2, "service_rate": 0.80},
        "success_threshold": 0.60,
        "shift_start": "11:00",
        "step_duration_minutes": 30,
        "total_steps": 12,
        "initial_rating": 4.3,
        "active_staff": ("Ravi", "Priya", "Sneha", "Vikram", "Arjun"),
        "demand_pattern": [1.0, 1.0, 1.2, 1.5, 1.5, 1.3, 1.0, 0.8, 0.7, 0.5, 0.3, 0.2],
        "special_events": {},
        "cost_multiplier": 1.0,
        "inventory_multiplier": 1.0,
    },
    "weekend_rush": {
        "name": "Weekend Festival Rush",
        "difficulty": "medium",
        "description": (
            "Demand spikes hard, the restaurant starts with a bruised rating, "
            "and a large party arrives mid-service."
        ),
        "targets": {"profit": 12000, "rating": 4.0, "service_rate": 0.75},
        "success_threshold": 0.60,
        "shift_start": "11:00",
        "step_duration_minutes": 30,
        "total_steps": 12,
        "initial_rating": 3.5,
        "active_staff": ("Priya", "Amit", "Sneha", "Vikram", "Meera", "Arjun", "Kavita"),
        "demand_pattern": [1.5, 1.8, 2.0, 2.5, 2.5, 2.2, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8],
        "special_events": {4: "large_party"},
        "cost_multiplier": 1.0,
        "inventory_multiplier": 1.0,
    },
    "crisis_shift": {
        "name": "Crisis Management Shift",
        "difficulty": "hard",
        "description": (
            "Costs are inflated, inventory is thin, demand is softer, and a health "
            "inspection forces the agent to protect compliance while staying solvent."
        ),
        "targets": {"profit": 5000, "rating": 4.0, "service_rate": 0.70},
        "success_threshold": 0.60,
        "shift_start": "11:00",
        "step_duration_minutes": 30,
        "total_steps": 12,
        "initial_rating": 4.0,
        "active_staff": ("Ravi", "Sneha", "Arjun"),
        "demand_pattern": [0.8, 0.9, 1.0, 1.2, 1.3, 1.2, 1.0, 0.9, 0.8, 0.7, 0.5, 0.3],
        "special_events": {8: "health_inspection"},
        "cost_multiplier": 2.0,
        "inventory_multiplier": 0.4,
    },
    "monsoon_delivery_crunch": {
        "name": "Monsoon Delivery Crunch",
        "difficulty": "medium",
        "description": (
            "Heavy rain shifts demand toward fast comfort food, delivery windows tighten, "
            "and supply delays make selective menu management more valuable."
        ),
        "targets": {"profit": 9000, "rating": 4.1, "service_rate": 0.78},
        "success_threshold": 0.62,
        "shift_start": "18:00",
        "step_duration_minutes": 30,
        "total_steps": 12,
        "initial_rating": 4.1,
        "active_staff": ("Ravi", "Sneha", "Meera", "Arjun"),
        "demand_pattern": [0.9, 1.1, 1.4, 1.7, 1.9, 2.0, 1.8, 1.5, 1.3, 1.0, 0.8, 0.6],
        "special_events": {3: "delivery_surge", 6: "supplier_delay"},
        "cost_multiplier": 1.1,
        "inventory_multiplier": 0.8,
    },
    "wedding_catering_chaos": {
        "name": "Wedding Catering Chaos",
        "difficulty": "hard",
        "description": (
            "A lucrative pre-booked event creates a burst of premium demand, but any "
            "service collapse damages both margin and reputation."
        ),
        "targets": {"profit": 15000, "rating": 4.1, "service_rate": 0.82},
        "success_threshold": 0.65,
        "shift_start": "17:00",
        "step_duration_minutes": 30,
        "total_steps": 12,
        "initial_rating": 4.4,
        "active_staff": ("Ravi", "Priya", "Sneha", "Vikram", "Meera", "Arjun"),
        "demand_pattern": [0.8, 1.0, 1.2, 1.4, 1.8, 2.4, 2.8, 2.6, 2.0, 1.4, 1.0, 0.7],
        "special_events": {5: "vip_review", 6: "large_party"},
        "cost_multiplier": 1.15,
        "inventory_multiplier": 1.1,
    },
    "office_catering_lunch": {
        "name": "Office Catering Lunch",
        "difficulty": "easy",
        "description": (
            "A reliable corporate lunch block boosts midday volume, rewarding clean "
            "prep and disciplined staffing without major shocks."
        ),
        "targets": {"profit": 10000, "rating": 4.25, "service_rate": 0.84},
        "success_threshold": 0.63,
        "shift_start": "10:30",
        "step_duration_minutes": 30,
        "total_steps": 12,
        "initial_rating": 4.35,
        "active_staff": ("Ravi", "Priya", "Sneha", "Meera", "Arjun"),
        "demand_pattern": [0.9, 1.1, 1.4, 1.8, 2.1, 2.0, 1.7, 1.2, 0.9, 0.7, 0.5, 0.3],
        "special_events": {4: "large_party"},
        "cost_multiplier": 1.0,
        "inventory_multiplier": 1.0,
    },
    "tourist_season_dinner": {
        "name": "Tourist Season Dinner",
        "difficulty": "medium",
        "description": (
            "An evening tourist wave brings strong premium demand, but aggressive pricing "
            "or understaffing quickly damages reputation and repeat traffic."
        ),
        "targets": {"profit": 13000, "rating": 4.15, "service_rate": 0.8},
        "success_threshold": 0.64,
        "shift_start": "18:30",
        "step_duration_minutes": 30,
        "total_steps": 12,
        "initial_rating": 4.0,
        "active_staff": ("Ravi", "Priya", "Sneha", "Vikram", "Meera", "Arjun"),
        "demand_pattern": [1.0, 1.2, 1.6, 1.9, 2.2, 2.4, 2.3, 2.0, 1.7, 1.3, 0.9, 0.6],
        "special_events": {2: "vip_review", 5: "delivery_surge"},
        "cost_multiplier": 1.08,
        "inventory_multiplier": 0.95,
    },
    "staff_shortage_recovery": {
        "name": "Staff Shortage Recovery",
        "difficulty": "hard",
        "description": (
            "The shift opens under-staffed with weak inventory buffers, forcing the agent "
            "to recover service quality through selective call-ins and careful menu scope."
        ),
        "targets": {"profit": 7000, "rating": 4.0, "service_rate": 0.76},
        "success_threshold": 0.61,
        "shift_start": "12:00",
        "step_duration_minutes": 30,
        "total_steps": 12,
        "initial_rating": 3.8,
        "active_staff": ("Amit", "Sneha"),
        "demand_pattern": [1.1, 1.2, 1.4, 1.6, 1.7, 1.7, 1.5, 1.3, 1.0, 0.8, 0.6, 0.4],
        "special_events": {6: "supplier_delay", 8: "health_inspection"},
        "cost_multiplier": 1.12,
        "inventory_multiplier": 0.7,
    },
}


def build_task(task_id: str) -> dict[str, Any]:
    spec = TASK_SPECS[task_id]
    staff = _activate(_base_staff(), spec["active_staff"])
    return {
        "task_id": task_id,
        "task_name": spec["name"],
        "difficulty": spec["difficulty"],
        "description": spec["description"],
        "total_steps": spec["total_steps"],
        "shift_start": spec["shift_start"],
        "step_duration_minutes": spec["step_duration_minutes"],
        "menu": _base_menu(),
        "staff": staff,
        "inventory": _base_inventory(
            cost_multiplier=spec["cost_multiplier"],
            quantity_multiplier=spec["inventory_multiplier"],
        ),
        "initial_rating": spec["initial_rating"],
        "demand_pattern": list(spec["demand_pattern"]),
        "special_events": dict(spec["special_events"]),
        "targets": dict(spec["targets"]),
        "success_threshold": spec["success_threshold"],
    }


def list_task_metadata() -> list[dict[str, Any]]:
    return [
        {
            "id": task_id,
            "name": spec["name"],
            "difficulty": spec["difficulty"],
            "description": spec["description"],
            "targets": spec["targets"],
            "max_steps": spec["total_steps"],
            "success_threshold": spec["success_threshold"],
        }
        for task_id, spec in TASK_SPECS.items()
    ]


def get_task(task_id: str) -> dict[str, Any]:
    if task_id not in TASK_SPECS:
        raise KeyError(f"Unknown task '{task_id}'. Available: {list(TASK_SPECS.keys())}")
    return build_task(task_id)
