"""
Restaurant management environment.

Follows a reset/step loop:
  1. env.reset("weekday_lunch")  → returns initial RestaurantState
  2. env.step(AgentAction(...))  → returns (new_state, reward, done, info)
  3. repeat step() until done=True
  4. env.get_result()            → returns ShiftResult for grading
"""

from __future__ import annotations

import math

from env.models import (
    AgentAction,
    InventoryItem,
    MenuItem,
    RestaurantState,
    ShiftResult,
    StaffMember,
)
from env.tasks import get_task

# ── Constants ──────────────────────────────────────────────────────────

BASE_CUSTOMERS_PER_STEP = 10  # at demand=1.0, ~10 customers per 30 min
ORDERS_PER_CUSTOMER = 1.5     # each customer orders ~1.5 items on average
PROMOTION_DEMAND_BOOST = 1.3  # promotion increases demand by 30%
PROMOTION_DISCOUNT = 0.15     # promotion gives 15% discount on prices
LARGE_PARTY_SIZE = 15         # extra customers from large_party event
INSPECTION_RATING_PENALTY = 0.5  # rating penalty if kitchen is dirty during inspection
INSPECTION_CLEANLINESS_THRESHOLD = 0.6  # need dishwasher skill above this to pass


class RestaurantEnv:
    """
    The restaurant shift simulation environment.

    Usage:
        env = RestaurantEnv()
        state = env.reset("weekday_lunch")
        while not done:
            action = agent.decide(state)
            state, reward, done, info = env.step(action)
        result = env.get_result()
    """

    def __init__(self) -> None:
        # These will be set by reset()
        self._task_config: dict | None = None
        self._menu: list[MenuItem] = []
        self._staff: list[StaffMember] = []
        self._inventory: list[InventoryItem] = []
        self._step: int = 0
        self._total_steps: int = 0
        self._shift_start: str = ""
        self._step_duration_minutes: int = 30
        self._demand_pattern: list[float] = []
        self._special_events: dict[int, str] = {}

        # Cumulative shift metrics
        self._revenue: float = 0.0
        self._costs: float = 0.0
        self._completed_orders: int = 0
        self._failed_orders: int = 0
        self._customer_rating: float = 4.0
        self._rating_count: int = 0  # how many ratings contributed

        # Track if environment has been reset
        self._is_reset: bool = False

    def reset(self, task_id: str) -> RestaurantState:
        """
        Initialize the environment for a new shift.

        Args:
            task_id: one of "weekday_lunch", "weekend_rush", "crisis_shift"

        Returns:
            The initial RestaurantState observation.
        """
        config = get_task(task_id)
        self._task_config = config

        # Deep-copy models so we can mutate them without affecting the task definition
        self._menu = [item.model_copy(deep=True) for item in config["menu"]]
        self._staff = [member.model_copy(deep=True) for member in config["staff"]]
        self._inventory = [inv.model_copy(deep=True) for inv in config["inventory"]]

        # Shift timing
        self._step = 0
        self._total_steps = config["total_steps"]
        self._shift_start = config["shift_start"]
        self._step_duration_minutes = config["step_duration_minutes"]
        self._demand_pattern = config["demand_pattern"]
        self._special_events = config.get("special_events", {})

        # Reset metrics
        self._revenue = 0.0
        self._costs = 0.0
        self._completed_orders = 0
        self._failed_orders = 0
        self._customer_rating = config["initial_rating"]
        self._rating_count = 1  # start with 1 so initial rating has weight

        self._is_reset = True

        return self._build_state()

    # ═══════════════════════════════════════════════════════════════════
    # STEP — the core simulation loop
    # ═══════════════════════════════════════════════════════════════════

    def step(self, action: AgentAction) -> tuple[RestaurantState, float, bool, dict]:
        """
        Process one time step of the shift.

        Args:
            action: the AgentAction with the manager's decisions

        Returns:
            (state, reward, done, info) where:
              - state:  the new RestaurantState after this step
              - reward: a float score for this step (higher = better)
              - done:   True if the shift is over
              - info:   dict with extra details about what happened
        """
        if not self._is_reset:
            raise RuntimeError("Must call reset() before step().")

        if self._step >= self._total_steps:
            raise RuntimeError("Shift is already over. Call reset() to start a new one.")

        info: dict = {"step": self._step, "events": []}

        # ── Phase 1: Apply the agent's decisions ──────────────────────
        self._apply_staff_changes(action, info)
        self._apply_menu_changes(action, info)
        self._apply_price_adjustments(action, info)
        self._apply_inventory_reorders(action, info)

        # ── Phase 2: Calculate staff costs for this step ──────────────
        staff_cost = self._calculate_staff_cost()
        self._costs += staff_cost
        info["staff_cost"] = round(staff_cost, 2)

        # ── Phase 3: Determine how many customers arrive ──────────────
        demand = self._current_demand()
        if action.promotion_active:
            demand *= PROMOTION_DEMAND_BOOST
            info["events"].append("promotion_active")

        # Check for special events
        event = self._special_events.get(self._step)
        extra_customers = 0
        if event == "large_party":
            extra_customers = LARGE_PARTY_SIZE
            info["events"].append("large_party_arrived")
        elif event == "health_inspection":
            self._handle_health_inspection(info)

        num_customers = int(BASE_CUSTOMERS_PER_STEP * demand) + extra_customers
        info["customers_arrived"] = num_customers

        # ── Phase 4: Process orders ───────────────────────────────────
        total_orders = int(num_customers * ORDERS_PER_CUSTOMER)
        served, failed, order_revenue, food_cost = self._process_orders(
            total_orders, action.promotion_active
        )

        self._completed_orders += served
        self._failed_orders += failed
        self._revenue += order_revenue
        self._costs += food_cost

        info["orders_attempted"] = total_orders
        info["orders_served"] = served
        info["orders_failed"] = failed
        info["order_revenue"] = round(order_revenue, 2)
        info["food_cost"] = round(food_cost, 2)

        # ── Phase 5: Update customer rating ───────────────────────────
        step_rating = self._calculate_step_rating(served, failed, total_orders)
        self._update_rating(step_rating)
        info["step_rating"] = round(step_rating, 2)

        # ── Phase 6: Calculate reward for this step ───────────────────
        step_profit = order_revenue - staff_cost - food_cost
        reward = self._calculate_reward(step_profit, step_rating, served, failed)
        info["step_profit"] = round(step_profit, 2)
        info["reward"] = round(reward, 2)

        # ── Phase 7: Advance clock ────────────────────────────────────
        self._step += 1
        done = self._step >= self._total_steps

        state = self._build_state()
        return state, reward, done, info

    # ═══════════════════════════════════════════════════════════════════
    # GET RESULT — end-of-shift summary
    # ═══════════════════════════════════════════════════════════════════

    def get_result(self) -> ShiftResult:
        """
        Produce the end-of-shift summary for grading.

        Should be called after the last step() returns done=True.
        """
        if not self._is_reset:
            raise RuntimeError("Must call reset() and run the shift first.")

        profit = self._revenue - self._costs
        total_orders = self._completed_orders + self._failed_orders

        # Customer satisfaction: combination of rating + service success
        if total_orders > 0:
            service_rate = self._completed_orders / total_orders
        else:
            service_rate = 0.0

        # Satisfaction = 60% rating-based + 40% service-based, scaled to 0-100
        rating_score = (self._customer_rating - 1.0) / 4.0  # normalize 1-5 to 0-1
        satisfaction = (0.6 * rating_score + 0.4 * service_rate) * 100

        return ShiftResult(
            total_revenue=round(self._revenue, 2),
            total_costs=round(self._costs, 2),
            profit=round(profit, 2),
            average_rating=round(self._customer_rating, 2),
            orders_served=self._completed_orders,
            orders_failed=self._failed_orders,
            customer_satisfaction=round(min(max(satisfaction, 0), 100), 2),
        )

    # ═══════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS — applying actions
    # ═══════════════════════════════════════════════════════════════════

    def _apply_staff_changes(self, action: AgentAction, info: dict) -> None:
        """Activate or deactivate staff based on agent's decisions."""
        for staff in self._staff:
            if staff.name in action.staff_changes:
                old = staff.is_active
                staff.is_active = action.staff_changes[staff.name]
                if old != staff.is_active:
                    status = "called_in" if staff.is_active else "sent_home"
                    info["events"].append(f"{staff.name}_{status}")

    def _apply_menu_changes(self, action: AgentAction, info: dict) -> None:
        """Enable or disable menu items based on agent's decisions."""
        for item in self._menu:
            if item.name in action.menu_changes:
                old = item.available
                item.available = action.menu_changes[item.name]
                if old != item.available:
                    status = "enabled" if item.available else "disabled"
                    info["events"].append(f"{item.name}_{status}")

    def _apply_price_adjustments(self, action: AgentAction, info: dict) -> None:
        """Change menu item prices based on agent's decisions."""
        for item in self._menu:
            if item.name in action.price_adjustments:
                new_price = action.price_adjustments[item.name]
                if new_price > 0:
                    old_price = item.price
                    item.price = new_price
                    info["events"].append(
                        f"{item.name}_price_{old_price}->{new_price}"
                    )

    def _apply_inventory_reorders(self, action: AgentAction, info: dict) -> None:
        """
        Process emergency inventory reorders.

        The agent pays reorder_cost_per_unit (more expensive than normal)
        and inventory is added immediately.
        """
        for inv in self._inventory:
            if inv.name in action.reorder_inventory:
                qty = action.reorder_inventory[inv.name]
                if qty > 0:
                    cost = qty * inv.reorder_cost_per_unit
                    inv.quantity += qty
                    self._costs += cost
                    info["events"].append(
                        f"reordered_{inv.name}_{qty}_{inv.unit}_cost_{cost:.0f}"
                    )

    # ═══════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS — simulation
    # ═══════════════════════════════════════════════════════════════════

    def _calculate_staff_cost(self) -> float:
        """Calculate wage cost for all active staff for one step (30 min = 0.5 hours)."""
        hours = self._step_duration_minutes / 60.0
        return sum(
            s.hourly_wage * hours
            for s in self._staff
            if s.is_active
        )

    def _get_kitchen_capacity(self) -> int:
        """
        How many orders the kitchen can handle this step.

        Based on active chefs and their skill levels.
        Each chef can handle ~5 orders per step, scaled by skill.
        """
        base_per_chef = 5
        capacity = 0
        for s in self._staff:
            if s.is_active and s.role == "chef":
                capacity += int(base_per_chef * (0.5 + 0.5 * s.skill_level))
                # skill 0.0 → 2 orders, skill 0.5 → 3, skill 1.0 → 5
        return max(capacity, 0)

    def _get_server_capacity(self) -> int:
        """
        How many orders servers can deliver this step.

        Each server handles ~8 orders per step, scaled by skill.
        """
        base_per_server = 8
        capacity = 0
        for s in self._staff:
            if s.is_active and s.role == "server":
                capacity += int(base_per_server * (0.5 + 0.5 * s.skill_level))
        return max(capacity, 0)

    def _check_ingredients(self, item: MenuItem) -> bool:
        """Check if we have enough ingredients to make one serving of this item."""
        for ingredient_name, qty_needed in item.ingredients.items():
            inv = self._find_inventory(ingredient_name)
            if inv is None or inv.quantity < qty_needed:
                return False
        return True

    def _consume_ingredients(self, item: MenuItem) -> float:
        """
        Deduct ingredients for one serving. Returns the ingredient cost.
        """
        cost = 0.0
        for ingredient_name, qty_needed in item.ingredients.items():
            inv = self._find_inventory(ingredient_name)
            if inv is not None:
                inv.quantity = round(inv.quantity - qty_needed, 4)
                cost += qty_needed * inv.cost_per_unit
        return cost

    def _find_inventory(self, name: str) -> InventoryItem | None:
        """Find an inventory item by name."""
        for inv in self._inventory:
            if inv.name == name:
                return inv
        return None

    def _process_orders(
        self, total_orders: int, promotion_active: bool
    ) -> tuple[int, int, float, float]:
        """
        Attempt to serve orders. Returns (served, failed, revenue, food_cost).

        Orders are limited by:
        1. Kitchen capacity (chefs)
        2. Server capacity
        3. Ingredient availability
        4. Menu item availability
        """
        kitchen_cap = self._get_kitchen_capacity()
        server_cap = self._get_server_capacity()
        service_cap = min(kitchen_cap, server_cap)

        # Get list of available menu items
        available_items = [item for item in self._menu if item.available]
        if not available_items:
            return 0, total_orders, 0.0, 0.0

        served = 0
        failed = 0
        revenue = 0.0
        food_cost = 0.0

        for i in range(total_orders):
            # Reached staff capacity — all remaining orders fail
            if served >= service_cap:
                failed += total_orders - i
                break

            # Pick a menu item (deterministic: cycle through available items)
            item = available_items[i % len(available_items)]

            # Check ingredients
            if not self._check_ingredients(item):
                failed += 1
                continue

            # Serve the order
            ingredient_cost = self._consume_ingredients(item)
            price = item.price
            if promotion_active:
                price = price * (1 - PROMOTION_DISCOUNT)

            revenue += price
            food_cost += ingredient_cost
            served += 1

        return served, failed, round(revenue, 2), round(food_cost, 2)

    def _calculate_step_rating(
        self, served: int, failed: int, total: int
    ) -> float:
        """
        Calculate a rating (1-5) for this step based on service quality.

        Factors:
        - Service success rate (did we serve all orders?)
        - Chef skill (higher skill = better food)
        - Server skill (higher skill = better service)
        """
        if total == 0:
            return self._customer_rating  # no customers = no change

        # Base: service success rate (0 to 1)
        success_rate = served / total if total > 0 else 0

        # Chef quality bonus (average skill of active chefs)
        active_chefs = [s for s in self._staff if s.is_active and s.role == "chef"]
        chef_skill = (
            sum(c.skill_level for c in active_chefs) / len(active_chefs)
            if active_chefs
            else 0.0
        )

        # Server quality bonus (average skill of active servers)
        active_servers = [s for s in self._staff if s.is_active and s.role == "server"]
        server_skill = (
            sum(s.skill_level for s in active_servers) / len(active_servers)
            if active_servers
            else 0.0
        )

        # Rating formula: 1-5 scale
        # 50% success rate + 25% chef quality + 25% server quality
        raw = success_rate * 0.5 + chef_skill * 0.25 + server_skill * 0.25
        rating = 1.0 + raw * 4.0  # scale 0-1 → 1-5

        return round(min(max(rating, 1.0), 5.0), 2)

    def _update_rating(self, new_rating: float) -> None:
        """Update the running average customer rating."""
        self._rating_count += 1
        # Weighted moving average: new ratings matter more than old ones
        weight = 0.3  # new rating gets 30% weight
        self._customer_rating = (
            (1 - weight) * self._customer_rating + weight * new_rating
        )
        self._customer_rating = min(max(self._customer_rating, 1.0), 5.0)

    def _handle_health_inspection(self, info: dict) -> None:
        """
        Handle a health inspection event.

        If there's no active dishwasher with sufficient skill,
        the rating takes a penalty.
        """
        active_dishwashers = [
            s for s in self._staff
            if s.is_active and s.role == "dishwasher"
        ]
        if not active_dishwashers:
            # No dishwasher = fail inspection
            self._customer_rating = max(
                1.0, self._customer_rating - INSPECTION_RATING_PENALTY
            )
            info["events"].append("health_inspection_FAILED_no_dishwasher")
        else:
            best_skill = max(d.skill_level for d in active_dishwashers)
            if best_skill >= INSPECTION_CLEANLINESS_THRESHOLD:
                info["events"].append("health_inspection_PASSED")
            else:
                penalty = INSPECTION_RATING_PENALTY * 0.5  # partial fail
                self._customer_rating = max(
                    1.0, self._customer_rating - penalty
                )
                info["events"].append("health_inspection_PARTIAL_FAIL")

    def _calculate_reward(
        self, profit: float, rating: float, served: int, failed: int
    ) -> float:
        """
        Calculate the reward signal for this step.

        Reward = weighted combination of:
          - Profit (normalized) — 40%
          - Rating quality      — 30%
          - Service success     — 30%

        This is NOT the final grade. It's a per-step signal the agent
        can use to learn which actions are working.
        """
        # Profit component: normalize to roughly -1 to +1 range
        profit_score = math.tanh(profit / 5000)  # tanh keeps it bounded

        # Rating component: 1-5 → 0-1
        rating_score = (rating - 1.0) / 4.0

        # Service component
        total = served + failed
        service_score = served / total if total > 0 else 0.5

        reward = (
            0.4 * profit_score
            + 0.3 * rating_score
            + 0.3 * service_score
        )
        return round(reward, 4)

    # ═══════════════════════════════════════════════════════════════════
    # OBSERVATION HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def _current_time(self) -> str:
        """Calculate human-readable time for the current step."""
        hours, minutes = map(int, self._shift_start.split(":"))
        total_minutes = hours * 60 + minutes + self._step * self._step_duration_minutes
        h = (total_minutes // 60) % 24
        m = total_minutes % 60
        return f"{h:02d}:{m:02d}"

    def _current_demand(self) -> float:
        """Get the demand level for the current step."""
        if self._step < len(self._demand_pattern):
            return self._demand_pattern[self._step]
        return 0.5  # fallback: low demand

    def _build_state(self) -> RestaurantState:
        """Build the current RestaurantState observation."""
        return RestaurantState(
            step=self._step,
            total_steps=self._total_steps,
            time_of_day=self._current_time(),
            menu=[item.model_copy(deep=True) for item in self._menu],
            staff=[member.model_copy(deep=True) for member in self._staff],
            inventory=[inv.model_copy(deep=True) for inv in self._inventory],
            active_orders=0,
            completed_orders=self._completed_orders,
            failed_orders=self._failed_orders,
            revenue=self._revenue,
            costs=self._costs,
            customer_rating=round(self._customer_rating, 2),
            demand_level=self._current_demand(),
            pending_customers=0,
        )