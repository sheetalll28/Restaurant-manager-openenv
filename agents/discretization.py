"""
State and action discretization for tabular Q-learning.

The full RestaurantState / AgentAction space is high-dimensional and structured.
This module maps observations to a compact tuple key and a finite set of macro-actions
that cover the main management levers: staffing, inventory, menu, and promotions.
"""

from __future__ import annotations

from env.models import AgentAction, RestaurantState

# Macro-action indices (used as columns in the Q-table)
ACTION_NOOP = 0
ACTION_STAFF_UP = 1
ACTION_STAFF_DOWN = 2
ACTION_REORDER = 3
ACTION_MENU_FIX = 4
ACTION_PROMOTE = 5
ACTION_ENSURE_DISHWASHER = 6
ACTION_FULL_RESPONSE = 7

ACTION_COUNT = 8
ACTION_NAMES = (
    "noop",
    "staff_up",
    "staff_down",
    "reorder",
    "menu_fix",
    "promote",
    "ensure_dishwasher",
    "full_response",
)

LOW_STOCK_THRESHOLD = 5.0
REORDER_AMOUNT = 10.0


def _demand_bin(demand: float) -> int:
    if demand < 0.8:
        return 0
    if demand < 1.3:
        return 1
    return 2


def _rating_bin(rating: float) -> int:
    if rating < 3.5:
        return 0
    if rating < 4.0:
        return 1
    return 2


def _stock_bin(state: RestaurantState) -> int:
    low_stock = sum(1 for inv in state.inventory if inv.quantity < LOW_STOCK_THRESHOLD)
    if low_stock == 0:
        return 0
    if low_stock <= 2:
        return 1
    return 2


def _phase_bin(state: RestaurantState) -> int:
    if state.total_steps <= 1:
        return 0
    ratio = state.step / max(state.total_steps - 1, 1)
    if ratio < 0.33:
        return 0
    if ratio < 0.67:
        return 1
    return 2


def encode_state(state: RestaurantState) -> tuple[int, ...]:
    """Map a RestaurantState observation to a hashable discrete state key."""
    active_chefs = sum(1 for member in state.staff if member.is_active and member.role == "chef")
    active_servers = sum(
        1 for member in state.staff if member.is_active and member.role == "server"
    )
    has_dishwasher = any(
        member.is_active and member.role == "dishwasher" for member in state.staff
    )

    return (
        _demand_bin(state.demand_level),
        _rating_bin(state.customer_rating),
        _stock_bin(state),
        min(active_chefs, 2),
        min(active_servers, 2),
        1 if has_dishwasher else 0,
        _phase_bin(state),
    )


def _staff_by_role(state: RestaurantState, role: str) -> list:
    return [member for member in state.staff if member.role == role]


def _apply_staff_up(action: AgentAction, state: RestaurantState) -> None:
    for member in _staff_by_role(state, "chef"):
        if not member.is_active:
            action.staff_changes[member.name] = True
    for member in _staff_by_role(state, "server"):
        if not member.is_active:
            action.staff_changes[member.name] = True


def _apply_staff_down(action: AgentAction, state: RestaurantState) -> None:
    for role in ("chef", "server"):
        members = sorted(_staff_by_role(state, role), key=lambda s: s.skill_level, reverse=True)
        for index, member in enumerate(members):
            should_be_active = index == 0
            if member.is_active != should_be_active:
                action.staff_changes[member.name] = should_be_active


def _apply_reorder(action: AgentAction, state: RestaurantState) -> None:
    for inv in state.inventory:
        if inv.quantity < LOW_STOCK_THRESHOLD:
            action.reorder_inventory[inv.name] = REORDER_AMOUNT


def _apply_menu_fix(action: AgentAction, state: RestaurantState) -> None:
    inv_lookup = {inv.name: inv.quantity for inv in state.inventory}
    for item in state.menu:
        can_make = all(
            inv_lookup.get(ingredient, 0) >= qty_needed * 3
            for ingredient, qty_needed in item.ingredients.items()
        )
        if not can_make and item.available:
            action.menu_changes[item.name] = False
        elif can_make and not item.available:
            action.menu_changes[item.name] = True


def _apply_ensure_dishwasher(action: AgentAction, state: RestaurantState) -> None:
    dishwashers = _staff_by_role(state, "dishwasher")
    if any(member.is_active for member in dishwashers):
        return
    best = max(dishwashers, key=lambda member: member.skill_level)
    action.staff_changes[best.name] = True


def decode_action(action_index: int, state: RestaurantState) -> AgentAction:
    """Convert a discrete macro-action index into a structured AgentAction."""
    action = AgentAction()

    if action_index == ACTION_NOOP:
        return action

    if action_index == ACTION_STAFF_UP:
        _apply_staff_up(action, state)
        return action

    if action_index == ACTION_STAFF_DOWN:
        _apply_staff_down(action, state)
        return action

    if action_index == ACTION_REORDER:
        _apply_reorder(action, state)
        return action

    if action_index == ACTION_MENU_FIX:
        _apply_menu_fix(action, state)
        return action

    if action_index == ACTION_PROMOTE:
        if state.demand_level < 0.9 and state.customer_rating >= 3.5:
            action.promotion_active = True
        return action

    if action_index == ACTION_ENSURE_DISHWASHER:
        _apply_ensure_dishwasher(action, state)
        return action

    if action_index == ACTION_FULL_RESPONSE:
        _apply_staff_up(action, state)
        _apply_reorder(action, state)
        _apply_menu_fix(action, state)
        _apply_ensure_dishwasher(action, state)
        return action

    raise ValueError(f"Unknown action index {action_index}. Expected 0..{ACTION_COUNT - 1}.")
