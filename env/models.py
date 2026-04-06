"""
Core data models for the restaurant management environment.

All models use Pydantic v2 for validation and serialization.
These types are imported by every other module in the project.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Domain enums (using Literal for simplicity, no extra imports needed)
# ---------------------------------------------------------------------------

StaffRole = Literal["chef", "server", "dishwasher"]
MenuCategory = Literal["appetizer", "main", "dessert", "drink"]


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------


class MenuItem(BaseModel):
    """A dish or drink on the restaurant menu."""

    name: str
    price: float = Field(gt=0, description="Selling price to customer")
    cost: float = Field(gt=0, description="Ingredient cost to prepare")
    prep_time_minutes: int = Field(gt=0, description="Minutes to prepare")
    category: MenuCategory
    ingredients: dict[str, float] = Field(
        description="Ingredient name -> quantity needed per serving"
    )
    available: bool = Field(default=True, description="Currently on the menu?")


class StaffMember(BaseModel):
    """A restaurant employee who can be called in or sent home."""

    name: str
    role: StaffRole
    skill_level: float = Field(ge=0.0, le=1.0, description="0=novice, 1=expert")
    hourly_wage: float = Field(gt=0)
    is_active: bool = Field(default=False, description="Currently working this shift?")


class InventoryItem(BaseModel):
    """An ingredient in the restaurant's stock."""

    name: str
    quantity: float = Field(ge=0, description="Current stock level")
    unit: str = Field(description="e.g. kg, liters, pieces")
    cost_per_unit: float = Field(gt=0, description="Normal purchase cost")
    reorder_cost_per_unit: float = Field(
        gt=0, description="Emergency mid-shift reorder cost (usually higher)"
    )


# ---------------------------------------------------------------------------
# Agent action — what the AI decides each time step
# ---------------------------------------------------------------------------


class AgentAction(BaseModel):
    """
    The action an AI agent submits at each time step.

    Every field has a sensible default so the agent only needs to specify
    the things it wants to change.
    """

    # Staff: map staff name -> True (call in) / False (send home)
    staff_changes: dict[str, bool] = Field(
        default_factory=dict,
        description="Staff name -> should be active? Only include changes.",
    )

    # Menu: map item name -> True (enable) / False (disable)
    menu_changes: dict[str, bool] = Field(
        default_factory=dict,
        description="Menu item name -> available? Only include changes.",
    )

    # Pricing: map item name -> new price
    price_adjustments: dict[str, float] = Field(
        default_factory=dict,
        description="Menu item name -> new selling price.",
    )

    # Inventory: map ingredient name -> quantity to emergency-order
    reorder_inventory: dict[str, float] = Field(
        default_factory=dict,
        description="Ingredient name -> quantity to order mid-shift.",
    )

    # Promotion: offer a 15% discount to attract more customers
    promotion_active: bool = Field(
        default=False, description="Run a discount promotion this step?"
    )


# ---------------------------------------------------------------------------
# Restaurant state — the full observation the agent sees each step
# ---------------------------------------------------------------------------


class RestaurantState(BaseModel):
    """
    Complete observable state of the restaurant at one point in the shift.

    This is what the AI agent receives as input before deciding an action.
    """

    # Time
    step: int = Field(ge=0, description="Current step index (0-based)")
    total_steps: int = Field(gt=0, description="Total steps in the shift")
    time_of_day: str = Field(description="Human-readable time, e.g. '12:30'")

    # Restaurant composition
    menu: list[MenuItem]
    staff: list[StaffMember]
    inventory: list[InventoryItem]

    # Operational metrics (cumulative for the shift so far)
    active_orders: int = Field(ge=0, description="Orders being prepared right now")
    completed_orders: int = Field(ge=0, description="Successfully served so far")
    failed_orders: int = Field(ge=0, description="Orders we couldn't fulfill")
    revenue: float = Field(ge=0, description="Total revenue earned so far")
    costs: float = Field(ge=0, description="Total costs incurred so far")
    customer_rating: float = Field(
        ge=1.0, le=5.0, description="Running average rating (1-5 stars)"
    )

    # Demand context
    demand_level: float = Field(
        gt=0, description="Demand multiplier (1.0 = normal, 2.0 = double)"
    )
    pending_customers: int = Field(ge=0, description="Customers waiting to order")


# ---------------------------------------------------------------------------
# Shift result — summary produced at the end for grading
# ---------------------------------------------------------------------------


class ShiftResult(BaseModel):
    """End-of-shift summary used by the grader to score the agent."""

    total_revenue: float
    total_costs: float
    profit: float
    average_rating: float = Field(ge=1.0, le=5.0)
    orders_served: int = Field(ge=0)
    orders_failed: int = Field(ge=0)
    customer_satisfaction: float = Field(
        ge=0, le=100, description="Overall satisfaction score 0-100"
    )
