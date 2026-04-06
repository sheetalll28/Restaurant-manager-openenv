"""
Restaurant management environment.

Follows a reset/step loop:
  1. env.reset("weekday_lunch")  → returns initial RestaurantState
  2. env.step(AgentAction(...))  → returns (new_state, reward, done, info)
  3. repeat step() until done=True
  4. env.get_result()            → returns ShiftResult for grading

"""

from __future__ import annotations

from env.models import (
    AgentAction,
    InventoryItem,
    MenuItem,
    RestaurantState,
    ShiftResult,
    StaffMember,
)
from env.tasks import get_task


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

    # ── Helper methods ─────────────────────────────────────────────────

    def _current_time(self) -> str:
        """Calculate human-readable time for the current step."""
        # Parse start time
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

    