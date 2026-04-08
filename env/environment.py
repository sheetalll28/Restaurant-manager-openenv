from __future__ import annotations
import math
from env.models import AgentAction, InventoryItem, MenuItem, RestaurantState, ShiftResult, StaffMember
from env.tasks import get_task

BASE_CUSTOMERS_PER_STEP = 10
ORDERS_PER_CUSTOMER = 1.5
PROMOTION_DEMAND_BOOST = 1.3
PROMOTION_DISCOUNT = 0.15
LARGE_PARTY_SIZE = 15
INSPECTION_RATING_PENALTY = 0.5
INSPECTION_CLEANLINESS_THRESHOLD = 0.6

class RestaurantEnv:
    def __init__(self) -> None:
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
        self._revenue: float = 0.0
        self._costs: float = 0.0
        self._completed_orders: int = 0
        self._failed_orders: int = 0
        self._customer_rating: float = 4.0
        self._rating_count: int = 0
        self._is_reset: bool = False

    def reset(self, task_id: str) -> RestaurantState:
        config = get_task(task_id)
        self._task_config = config
        self._menu = [item.model_copy(deep=True) for item in config["menu"]]
        self._staff = [member.model_copy(deep=True) for member in config["staff"]]
        self._inventory = [inv.model_copy(deep=True) for inv in config["inventory"]]
        self._step = 0
        self._total_steps = config["total_steps"]
        self._shift_start = config["shift_start"]
        self._step_duration_minutes = config["step_duration_minutes"]
        self._demand_pattern = config["demand_pattern"]
        self._special_events = config.get("special_events", {})
        self._revenue = 0.0
        self._costs = 0.0
        self._completed_orders = 0
        self._failed_orders = 0
        self._customer_rating = config["initial_rating"]
        self._rating_count = 1
        self._is_reset = True
        return self._build_state()

    def step(self, action: AgentAction) -> tuple[RestaurantState, float, bool, dict]:
        if not self._is_reset:
            raise RuntimeError("Must call reset() before step().")
        if self._step >= self._total_steps:
            raise RuntimeError("Shift is already over. Call reset() to start a new one.")

        info: dict = {"step": self._step, "events": []}
        self._apply_staff_changes(action, info)
        self._apply_menu_changes(action, info)
        self._apply_price_adjustments(action, info)
        self._apply_inventory_reorders(action, info)

        staff_cost = self._calculate_staff_cost()
        self._costs += staff_cost
        info["staff_cost"] = round(staff_cost, 2)

        demand = self._current_demand()
        if action.promotion_active:
            demand *= PROMOTION_DEMAND_BOOST
            info["events"].append("promotion_active")

        event = self._special_events.get(self._step)
        extra_customers = 0
        if event == "large_party":
            extra_customers = LARGE_PARTY_SIZE
            info["events"].append("large_party_arrived")
        elif event == "health_inspection":
            self._handle_health_inspection(info)

        num_customers = int(BASE_CUSTOMERS_PER_STEP * demand) + extra_customers
        info["customers_arrived"] = num_customers

        total_orders = int(num_customers * ORDERS_PER_CUSTOMER)
        served, failed, order_revenue, food_cost = self._process_orders(total_orders, action.promotion_active)

        self._completed_orders += served
        self._failed_orders += failed
        self._revenue += order_revenue
        self._costs += food_cost

        info["orders_attempted"] = total_orders
        info["orders_served"] = served
        info["orders_failed"] = failed
        info["order_revenue"] = round(order_revenue, 2)
        info["food_cost"] = round(food_cost, 2)

        step_rating = self._calculate_step_rating(served, failed, total_orders)
        self._update_rating(step_rating)
        info["step_rating"] = round(step_rating, 2)

        step_profit = order_revenue - staff_cost - food_cost
        reward = self._calculate_reward(step_profit, step_rating, served, failed)
        info["step_profit"] = round(step_profit, 2)
        info["reward"] = round(reward, 2)

        self._step += 1
        done = self._step >= self._total_steps
        state = self._build_state()
        return state, reward, done, info

    def state(self) -> RestaurantState:
        if not self._is_reset:
            raise RuntimeError("Must call reset() before state().")
        return self._build_state()

    def get_result(self) -> ShiftResult:
        if not self._is_reset:
            raise RuntimeError("Must call reset() and run the shift first.")
        profit = self._revenue - self._costs
        total_orders = self._completed_orders + self._failed_orders
        service_rate = self._completed_orders / total_orders if total_orders > 0 else 0.0
        rating_score = (self._customer_rating - 1.0) / 4.0
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

    def _apply_staff_changes(self, action: AgentAction, info: dict) -> None:
        for staff in self._staff:
            if staff.name in action.staff_changes:
                old = staff.is_active
                staff.is_active = action.staff_changes[staff.name]
                if old != staff.is_active:
                    status = "called_in" if staff.is_active else "sent_home"
                    info["events"].append(f"{staff.name}_{status}")

    def _apply_menu_changes(self, action: AgentAction, info: dict) -> None:
        for item in self._menu:
            if item.name in action.menu_changes:
                old = item.available
                item.available = action.menu_changes[item.name]
                if old != item.available:
                    status = "enabled" if item.available else "disabled"
                    info["events"].append(f"{item.name}_{status}")

    def _apply_price_adjustments(self, action: AgentAction, info: dict) -> None:
        for item in self._menu:
            if item.name in action.price_adjustments:
                new_price = action.price_adjustments[item.name]
                if new_price > 0:
                    old_price = item.price
                    item.price = new_price
                    info["events"].append(f"{item.name}_price_{old_price}->{new_price}")

    def _apply_inventory_reorders(self, action: AgentAction, info: dict) -> None:
        for inv in self._inventory:
            if inv.name in action.reorder_inventory:
                qty = action.reorder_inventory[inv.name]
                if qty > 0:
                    cost = qty * inv.reorder_cost_per_unit
                    inv.quantity += qty
                    self._costs += cost
                    info["events"].append(f"reordered_{inv.name}_{qty}_{inv.unit}_cost_{cost:.0f}")

    def _calculate_staff_cost(self) -> float:
        hours = self._step_duration_minutes / 60.0
        return sum(s.hourly_wage * hours for s in self._staff if s.is_active)

    def _get_kitchen_capacity(self) -> int:
        base_per_chef = 5
        capacity = 0
        for s in self._staff:
            if s.is_active and s.role == "chef":
                capacity += int(base_per_chef * (0.5 + 0.5 * s.skill_level))
        return max(capacity, 0)

    def _get_server_capacity(self) -> int:
        base_per_server = 8
        capacity = 0
        for s in self._staff:
            if s.is_active and s.role == "server":
                capacity += int(base_per_server * (0.5 + 0.5 * s.skill_level))
        return max(capacity, 0)

    def _check_ingredients(self, item: MenuItem) -> bool:
        for ingredient_name, qty_needed in item.ingredients.items():
            inv = self._find_inventory(ingredient_name)
            if inv is None or inv.quantity < qty_needed:
                return False
        return True

    def _consume_ingredients(self, item: MenuItem) -> float:
        cost = 0.0
        for ingredient_name, qty_needed in item.ingredients.items():
            inv = self._find_inventory(ingredient_name)
            if inv is not None:
                inv.quantity = round(inv.quantity - qty_needed, 4)
                cost += qty_needed * inv.cost_per_unit
        return cost

    def _find_inventory(self, name: str) -> InventoryItem | None:
        for inv in self._inventory:
            if inv.name == name:
                return inv
        return None

    def _process_orders(self, total_orders: int, promotion_active: bool) -> tuple[int, int, float, float]:
        kitchen_cap = self._get_kitchen_capacity()
        server_cap = self._get_server_capacity()
        service_cap = min(kitchen_cap, server_cap)
        available_items = [item for item in self._menu if item.available]
        if not available_items:
            return 0, total_orders, 0.0, 0.0

        served = 0
        failed = 0
        revenue = 0.0
        food_cost = 0.0

        for i in range(total_orders):
            if served >= service_cap:
                failed += total_orders - i
                break
            item = available_items[i % len(available_items)]
            if not self._check_ingredients(item):
                failed += 1
                continue
            ingredient_cost = self._consume_ingredients(item)
            price = item.price * (1 - PROMOTION_DISCOUNT) if promotion_active else item.price
            revenue += price
            food_cost += ingredient_cost
            served += 1

        return served, failed, round(revenue, 2), round(food_cost, 2)

    def _calculate_step_rating(self, served: int, failed: int, total: int) -> float:
        if total == 0:
            return self._customer_rating
        success_rate = served / total if total > 0 else 0
        active_chefs = [s for s in self._staff if s.is_active and s.role == "chef"]
        chef_skill = sum(c.skill_level for c in active_chefs) / len(active_chefs) if active_chefs else 0.0
        active_servers = [s for s in self._staff if s.is_active and s.role == "server"]
        server_skill = sum(s.skill_level for s in active_servers) / len(active_servers) if active_servers else 0.0
        raw = success_rate * 0.5 + chef_skill * 0.25 + server_skill * 0.25
        rating = 1.0 + raw * 4.0
        return round(min(max(rating, 1.0), 5.0), 2)

    def _update_rating(self, new_rating: float) -> None:
        self._rating_count += 1
        weight = 0.3
        self._customer_rating = (1 - weight) * self._customer_rating + weight * new_rating
        self._customer_rating = min(max(self._customer_rating, 1.0), 5.0)

    def _handle_health_inspection(self, info: dict) -> None:
        active_dishwashers = [s for s in self._staff if s.is_active and s.role == "dishwasher"]
        if not active_dishwashers:
            self._customer_rating = max(1.0, self._customer_rating - INSPECTION_RATING_PENALTY)
            info["events"].append("health_inspection_FAILED_no_dishwasher")
        else:
            best_skill = max(d.skill_level for d in active_dishwashers)
            if best_skill >= INSPECTION_CLEANLINESS_THRESHOLD:
                info["events"].append("health_inspection_PASSED")
            else:
                penalty = INSPECTION_RATING_PENALTY * 0.5
                self._customer_rating = max(1.0, self._customer_rating - penalty)
                info["events"].append("health_inspection_PARTIAL_FAIL")

    def _calculate_reward(self, profit: float, rating: float, served: int, failed: int) -> float:
        profit_score = math.tanh(profit / 5000)
        rating_score = (rating - 1.0) / 4.0
        total = served + failed
        if total > 0:
            service_score = served / total
            failure_rate = failed / total
        else:
            service_score = 0.5
            failure_rate = 0.0
        reward = 0.4 * profit_score + 0.3 * rating_score + 0.3 * service_score
        failure_penalty_multiplier = (1.0 - failure_rate) ** 1.5
        reward = reward * failure_penalty_multiplier
        return round(reward, 4)

    def _current_time(self) -> str:
        hours, minutes = map(int, self._shift_start.split(":"))
        total_minutes = hours * 60 + minutes + self._step * self._step_duration_minutes
        h = (total_minutes // 60) % 24
        m = total_minutes % 60
        return f"{h:02d}:{m:02d}"

    def _current_demand(self) -> float:
        if self._step < len(self._demand_pattern):
            return self._demand_pattern[self._step]
        return 0.5

    def _build_state(self) -> RestaurantState:
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