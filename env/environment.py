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

BASE_CUSTOMERS_PER_STEP = 10
ORDERS_PER_CUSTOMER = 1.5
PROMOTION_DEMAND_BOOST = 1.25
PROMOTION_DISCOUNT = 0.15
LARGE_PARTY_SIZE = 15
INSPECTION_RATING_PENALTY = 0.5
INSPECTION_CLEANLINESS_THRESHOLD = 0.6
MAX_PRICE_MULTIPLIER = 1.6
MIN_PRICE_MULTIPLIER = 0.7


class RestaurantEnv:
    def __init__(self) -> None:
        self._task_config: dict | None = None
        self._menu: list[MenuItem] = []
        self._staff: list[StaffMember] = []
        self._inventory: list[InventoryItem] = []
        self._baseline_prices: dict[str, float] = {}
        self._step: int = 0
        self._total_steps: int = 0
        self._shift_start: str = ""
        self._step_duration_minutes: int = 30
        self._demand_pattern: list[float] = []
        self._special_events: dict[int, str] = {}
        self._revenue: float = 0.0
        self._costs: float = 0.0
        self._labor_costs: float = 0.0
        self._food_costs: float = 0.0
        self._reorder_costs: float = 0.0
        self._completed_orders: int = 0
        self._failed_orders: int = 0
        self._stockout_failures: int = 0
        self._capacity_failures: int = 0
        self._customer_rating: float = 4.0
        self._rating_count: int = 0
        self._is_reset: bool = False

    def reset(self, task_id: str) -> RestaurantState:
        config = get_task(task_id)
        self._task_config = config
        self._menu = [item.model_copy(deep=True) for item in config["menu"]]
        self._baseline_prices = {item.name: item.price for item in self._menu}
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
        self._labor_costs = 0.0
        self._food_costs = 0.0
        self._reorder_costs = 0.0
        self._completed_orders = 0
        self._failed_orders = 0
        self._stockout_failures = 0
        self._capacity_failures = 0
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
        reorder_cost = self._apply_inventory_reorders(action, info)

        staff_cost = self._calculate_staff_cost()
        self._costs += staff_cost
        self._labor_costs += staff_cost
        info["staff_cost"] = round(staff_cost, 2)
        info["reorder_cost"] = round(reorder_cost, 2)

        demand = self._current_demand()
        demand *= self._rating_demand_multiplier()
        price_multiplier = self._pricing_demand_multiplier()
        demand *= price_multiplier
        info["demand_modifiers"] = {
            "rating": round(self._rating_demand_multiplier(), 3),
            "pricing": round(price_multiplier, 3),
        }

        if action.promotion_active:
            demand *= PROMOTION_DEMAND_BOOST
            info["events"].append("promotion_active")

        event_multiplier, extra_customers = self._handle_step_event(info)
        demand *= event_multiplier

        num_customers = max(0, int(BASE_CUSTOMERS_PER_STEP * demand) + extra_customers)
        info["customers_arrived"] = num_customers

        total_orders = int(num_customers * ORDERS_PER_CUSTOMER)
        service_stats = self._process_orders(total_orders, action.promotion_active)

        served = service_stats["served"]
        failed = service_stats["failed"]
        order_revenue = service_stats["revenue"]
        food_cost = service_stats["food_cost"]
        stockout_failures = service_stats["stockout_failures"]
        capacity_failures = service_stats["capacity_failures"]

        self._completed_orders += served
        self._failed_orders += failed
        self._stockout_failures += stockout_failures
        self._capacity_failures += capacity_failures
        self._revenue += order_revenue
        self._costs += food_cost
        self._food_costs += food_cost

        info["orders_attempted"] = total_orders
        info["orders_served"] = served
        info["orders_failed"] = failed
        info["stockout_failures"] = stockout_failures
        info["capacity_failures"] = capacity_failures
        info["order_revenue"] = round(order_revenue, 2)
        info["food_cost"] = round(food_cost, 2)

        step_profit = order_revenue - staff_cost - food_cost - reorder_cost
        step_rating, rating_penalties = self._calculate_step_rating(
            served=served,
            failed=failed,
            total=total_orders,
            stockout_failures=stockout_failures,
            capacity_failures=capacity_failures,
            action=action,
        )
        self._update_rating(step_rating)
        info["step_rating"] = round(step_rating, 2)
        info["rating_penalties"] = rating_penalties

        reward, reward_breakdown = self._calculate_reward(
            profit=step_profit,
            rating=step_rating,
            served=served,
            failed=failed,
            stockout_failures=stockout_failures,
            capacity_failures=capacity_failures,
            reorder_cost=reorder_cost,
            staff_cost=staff_cost,
        )
        info["step_profit"] = round(step_profit, 2)
        info["reward"] = round(reward, 4)
        info["reward_breakdown"] = reward_breakdown

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
        satisfaction = (0.55 * rating_score + 0.45 * service_rate) * 100
        return ShiftResult(
            total_revenue=round(self._revenue, 2),
            total_costs=round(self._costs, 2),
            profit=round(profit, 2),
            average_rating=round(self._customer_rating, 2),
            orders_served=self._completed_orders,
            orders_failed=self._failed_orders,
            customer_satisfaction=round(min(max(satisfaction, 0), 100), 2),
            labor_costs=round(self._labor_costs, 2),
            food_costs=round(self._food_costs, 2),
            reorder_costs=round(self._reorder_costs, 2),
            stockout_failures=self._stockout_failures,
            capacity_failures=self._capacity_failures,
            service_rate=round(service_rate, 4),
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
            if item.name not in action.price_adjustments:
                continue
            new_price = action.price_adjustments[item.name]
            if new_price <= 0:
                continue
            base_price = self._baseline_prices.get(item.name, item.price)
            min_price = round(base_price * MIN_PRICE_MULTIPLIER, 2)
            max_price = round(base_price * MAX_PRICE_MULTIPLIER, 2)
            clamped_price = min(max(new_price, min_price), max_price)
            if clamped_price != new_price:
                info["events"].append(f"{item.name}_price_clamped")
            old_price = item.price
            item.price = clamped_price
            info["events"].append(f"{item.name}_price_{old_price}->{clamped_price}")

    def _apply_inventory_reorders(self, action: AgentAction, info: dict) -> float:
        reorder_cost = 0.0
        for inv in self._inventory:
            if inv.name not in action.reorder_inventory:
                continue
            qty = action.reorder_inventory[inv.name]
            if qty <= 0:
                continue
            cost = qty * inv.reorder_cost_per_unit
            inv.quantity += qty
            reorder_cost += cost
            self._costs += cost
            self._reorder_costs += cost
            info["events"].append(f"reordered_{inv.name}_{qty}_{inv.unit}_cost_{cost:.0f}")
        return reorder_cost

    def _calculate_staff_cost(self) -> float:
        hours = self._step_duration_minutes / 60.0
        return sum(staff.hourly_wage * hours for staff in self._staff if staff.is_active)

    def _get_kitchen_capacity(self) -> int:
        capacity = 0
        for staff in self._staff:
            if staff.is_active and staff.role == "chef":
                capacity += int(5 * (0.5 + 0.6 * staff.skill_level))
        return max(capacity, 0)

    def _get_server_capacity(self) -> int:
        capacity = 0
        for staff in self._staff:
            if staff.is_active and staff.role == "server":
                capacity += int(8 * (0.5 + 0.55 * staff.skill_level))
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

    def _process_orders(self, total_orders: int, promotion_active: bool) -> dict[str, float | int]:
        kitchen_cap = self._get_kitchen_capacity()
        server_cap = self._get_server_capacity()
        service_cap = min(kitchen_cap, server_cap)
        available_items = [item for item in self._menu if item.available]
        if not available_items:
            return {
                "served": 0,
                "failed": total_orders,
                "revenue": 0.0,
                "food_cost": 0.0,
                "stockout_failures": total_orders,
                "capacity_failures": 0,
            }

        served = 0
        failed = 0
        revenue = 0.0
        food_cost = 0.0
        stockout_failures = 0
        capacity_failures = 0

        for i in range(total_orders):
            if served >= service_cap:
                missed = total_orders - i
                failed += missed
                capacity_failures += missed
                break

            item = available_items[i % len(available_items)]
            if not self._check_ingredients(item):
                failed += 1
                stockout_failures += 1
                continue

            ingredient_cost = self._consume_ingredients(item)
            price = item.price * (1 - PROMOTION_DISCOUNT) if promotion_active else item.price
            revenue += price
            food_cost += ingredient_cost
            served += 1

        return {
            "served": served,
            "failed": failed,
            "revenue": round(revenue, 2),
            "food_cost": round(food_cost, 2),
            "stockout_failures": stockout_failures,
            "capacity_failures": capacity_failures,
        }

    def _calculate_step_rating(
        self,
        *,
        served: int,
        failed: int,
        total: int,
        stockout_failures: int,
        capacity_failures: int,
        action: AgentAction,
    ) -> tuple[float, dict[str, float]]:
        if total == 0:
            return self._customer_rating, {"service": 0.0, "stockout": 0.0, "pricing": 0.0}

        success_rate = served / total
        stockout_rate = stockout_failures / total
        capacity_rate = capacity_failures / total
        active_chefs = [staff for staff in self._staff if staff.is_active and staff.role == "chef"]
        chef_skill = (
            sum(member.skill_level for member in active_chefs) / len(active_chefs)
            if active_chefs
            else 0.0
        )
        active_servers = [staff for staff in self._staff if staff.is_active and staff.role == "server"]
        server_skill = (
            sum(member.skill_level for member in active_servers) / len(active_servers)
            if active_servers
            else 0.0
        )
        pricing_penalty = max(0.0, self._average_price_ratio() - 1.15) * 0.7
        promotion_penalty = 0.05 if action.promotion_active and capacity_rate > 0.2 else 0.0

        base = 1.0 + (0.42 * success_rate + 0.22 * chef_skill + 0.16 * server_skill + 0.20) * 4.0
        service_penalty = capacity_rate * 1.25
        stockout_penalty = stockout_rate * 0.9
        total_penalty = service_penalty + stockout_penalty + pricing_penalty + promotion_penalty
        rating = base - total_penalty
        rating = round(min(max(rating, 1.0), 5.0), 2)
        return rating, {
            "service": round(service_penalty, 3),
            "stockout": round(stockout_penalty, 3),
            "pricing": round(pricing_penalty + promotion_penalty, 3),
        }

    def _update_rating(self, new_rating: float) -> None:
        self._rating_count += 1
        weight = 0.35
        self._customer_rating = (1 - weight) * self._customer_rating + weight * new_rating
        self._customer_rating = min(max(self._customer_rating, 1.0), 5.0)

    def _handle_step_event(self, info: dict) -> tuple[float, int]:
        event = self._special_events.get(self._step)
        if event is None:
            return 1.0, 0

        if event == "large_party":
            info["events"].append("large_party_arrived")
            return 1.0, LARGE_PARTY_SIZE
        if event == "health_inspection":
            self._handle_health_inspection(info)
            return 1.0, 0
        if event == "delivery_surge":
            info["events"].append("delivery_surge")
            return 1.18, 4
        if event == "supplier_delay":
            info["events"].append("supplier_delay")
            for inv in self._inventory:
                inv.quantity = round(inv.quantity * 0.92, 3)
            return 0.96, 0
        if event == "vip_review":
            info["events"].append("vip_review_visit")
            self._customer_rating = min(5.0, self._customer_rating + 0.05)
            return 1.08, 2

        info["events"].append(f"unhandled_event_{event}")
        return 1.0, 0

    def _handle_health_inspection(self, info: dict) -> None:
        active_dishwashers = [
            staff for staff in self._staff if staff.is_active and staff.role == "dishwasher"
        ]
        if not active_dishwashers:
            self._customer_rating = max(1.0, self._customer_rating - INSPECTION_RATING_PENALTY)
            info["events"].append("health_inspection_FAILED_no_dishwasher")
            return

        best_skill = max(dishwasher.skill_level for dishwasher in active_dishwashers)
        if best_skill >= INSPECTION_CLEANLINESS_THRESHOLD:
            info["events"].append("health_inspection_PASSED")
        else:
            penalty = INSPECTION_RATING_PENALTY * 0.5
            self._customer_rating = max(1.0, self._customer_rating - penalty)
            info["events"].append("health_inspection_PARTIAL_FAIL")

    def _calculate_reward(
        self,
        *,
        profit: float,
        rating: float,
        served: int,
        failed: int,
        stockout_failures: int,
        capacity_failures: int,
        reorder_cost: float,
        staff_cost: float,
    ) -> tuple[float, dict[str, float]]:
        total = served + failed
        service_score = served / total if total > 0 else 0.0
        failure_rate = failed / total if total > 0 else 0.0
        stockout_rate = stockout_failures / total if total > 0 else 0.0
        capacity_rate = capacity_failures / total if total > 0 else 0.0

        profit_signal = math.tanh(profit / 3500.0)
        rating_signal = ((rating - 1.0) / 4.0) * 2.0 - 1.0
        service_signal = service_score * 2.0 - 1.0

        penalty = (
            failure_rate * 0.7
            + stockout_rate * 0.4
            + capacity_rate * 0.35
            + min(reorder_cost / 2500.0, 0.35)
            + min(staff_cost / 1800.0, 0.2) * max(0.0, 0.55 - service_score)
        )

        reward = 0.4 * profit_signal + 0.25 * rating_signal + 0.35 * service_signal - penalty
        reward = round(max(-1.0, min(1.0, reward)), 4)
        return reward, {
            "profit_signal": round(profit_signal, 4),
            "rating_signal": round(rating_signal, 4),
            "service_signal": round(service_signal, 4),
            "penalty": round(penalty, 4),
        }

    def _average_price_ratio(self) -> float:
        enabled_items = [item for item in self._menu if item.available]
        if not enabled_items:
            return 1.0
        ratios = [
            item.price / max(self._baseline_prices.get(item.name, item.price), 1.0)
            for item in enabled_items
        ]
        return sum(ratios) / len(ratios)

    def _pricing_demand_multiplier(self) -> float:
        ratio = self._average_price_ratio()
        if ratio > 1.0:
            return max(0.72, 1.0 - (ratio - 1.0) * 0.75)
        return min(1.18, 1.0 + (1.0 - ratio) * 0.35)

    def _rating_demand_multiplier(self) -> float:
        normalized = (self._customer_rating - 3.0) / 2.0
        return min(1.15, max(0.8, 0.95 + normalized * 0.18))

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
        total_failed = self._failed_orders
        pending_customers = max(0, int(total_failed / max(ORDERS_PER_CUSTOMER, 1.0)))
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
            revenue=round(self._revenue, 2),
            costs=round(self._costs, 2),
            customer_rating=round(self._customer_rating, 2),
            demand_level=round(self._current_demand(), 2),
            pending_customers=pending_customers,
        )

