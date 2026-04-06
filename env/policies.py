"""
Baseline policies for the restaurant environment.

These are simple rule-based strategies (no AI/LLM).
They serve as benchmarks to compare against the AI agent.
"""

from __future__ import annotations

from env.models import AgentAction, RestaurantState


def do_nothing_policy(state: RestaurantState) -> AgentAction:
    """
    The laziest possible policy — does nothing every step.

    Useful as a minimum baseline: any real agent should beat this.
    """
    return AgentAction()


def simple_rule_policy(state: RestaurantState) -> AgentAction:
    """
    A reasonable rule-based policy that a human manager might use.

    Rules:
    1. STAFFING: Scale staff based on demand level
       - High demand (>1.3) → call in extra chefs and servers
       - Low demand (<0.7)  → send home unnecessary staff to save money
       - Always keep at least 1 dishwasher active

    2. INVENTORY: Reorder if any ingredient drops below 5 units
       - Order enough for 10 more units

    3. MENU: Disable items if we can't make them (out of ingredients)

    4. PRICING: No price changes (keep it simple)

    5. PROMOTIONS: Run promotion if demand is low and rating is okay
    """
    action = AgentAction()
    demand = state.demand_level

    # ── Rule 1: Staffing decisions ────────────────────────────────────

    # Build lookup of staff by name for convenience
    staff_by_name = {s.name: s for s in state.staff}

    # Count current active staff by role
    active_chefs = [s for s in state.staff if s.is_active and s.role == "chef"]
    active_servers = [s for s in state.staff if s.is_active and s.role == "server"]
    active_dishwashers = [s for s in state.staff if s.is_active and s.role == "dishwasher"]

    # All staff by role (active or not)
    all_chefs = [s for s in state.staff if s.role == "chef"]
    all_servers = [s for s in state.staff if s.role == "server"]
    all_dishwashers = [s for s in state.staff if s.role == "dishwasher"]

    if demand > 1.3:
        # HIGH DEMAND: call in all chefs and servers
        for chef in all_chefs:
            if not chef.is_active:
                action.staff_changes[chef.name] = True
        for server in all_servers:
            if not server.is_active:
                action.staff_changes[server.name] = True
    elif demand < 0.7:
        # LOW DEMAND: keep only the best chef and best server
        # Sort by skill (highest first) and keep only the top one
        chefs_sorted = sorted(all_chefs, key=lambda s: s.skill_level, reverse=True)
        for i, chef in enumerate(chefs_sorted):
            if i == 0:
                # Keep the best chef active
                if not chef.is_active:
                    action.staff_changes[chef.name] = True
            else:
                # Send home the rest
                if chef.is_active:
                    action.staff_changes[chef.name] = False

        servers_sorted = sorted(all_servers, key=lambda s: s.skill_level, reverse=True)
        for i, server in enumerate(servers_sorted):
            if i == 0:
                if not server.is_active:
                    action.staff_changes[server.name] = True
            else:
                if server.is_active:
                    action.staff_changes[server.name] = False

    # Always ensure at least 1 dishwasher is active (for health inspection!)
    if not active_dishwashers:
        # Activate the best dishwasher
        best_dw = max(all_dishwashers, key=lambda s: s.skill_level)
        action.staff_changes[best_dw.name] = True

    # ── Rule 2: Inventory reorders ────────────────────────────────────

    LOW_STOCK_THRESHOLD = 5.0
    REORDER_AMOUNT = 10.0

    for inv in state.inventory:
        if inv.quantity < LOW_STOCK_THRESHOLD:
            action.reorder_inventory[inv.name] = REORDER_AMOUNT

    # ── Rule 3: Menu availability ─────────────────────────────────────

    # Build a quick inventory lookup
    inv_lookup = {inv.name: inv.quantity for inv in state.inventory}

    for item in state.menu:
        # Check if we have enough ingredients for at least 3 servings
        can_make = True
        for ingredient, qty_needed in item.ingredients.items():
            available = inv_lookup.get(ingredient, 0)
            if available < qty_needed * 3:
                can_make = False
                break

        if not can_make and item.available:
            action.menu_changes[item.name] = False  # disable it
        elif can_make and not item.available:
            action.menu_changes[item.name] = True  # re-enable it

    # ── Rule 4: Promotions ────────────────────────────────────────────

    # Run promotion if demand is low but we have good ratings
    if demand < 0.8 and state.customer_rating >= 3.5:
        action.promotion_active = True

    return action
