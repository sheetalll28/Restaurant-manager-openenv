from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

StaffRole = Literal["chef", "server", "dishwasher"]
MenuCategory = Literal["appetizer", "main", "dessert", "drink"]

class MenuItem(BaseModel):
    name: str
    price: float = Field(gt=0)
    cost: float = Field(gt=0)
    prep_time_minutes: int = Field(gt=0)
    category: MenuCategory
    ingredients: dict[str, float] = Field(default_factory=dict)
    available: bool = Field(default=True)

class StaffMember(BaseModel):
    name: str
    role: StaffRole
    skill_level: float = Field(ge=0.0, le=1.0)
    hourly_wage: float = Field(gt=0)
    is_active: bool = Field(default=False)

class InventoryItem(BaseModel):
    name: str
    quantity: float = Field(ge=0)
    unit: str
    cost_per_unit: float = Field(gt=0)
    reorder_cost_per_unit: float = Field(gt=0)

class AgentAction(BaseModel):
    staff_changes: dict[str, bool] = Field(default_factory=dict)
    menu_changes: dict[str, bool] = Field(default_factory=dict)
    price_adjustments: dict[str, float] = Field(default_factory=dict)
    reorder_inventory: dict[str, float] = Field(default_factory=dict)
    promotion_active: bool = Field(default=False)

class RestaurantState(BaseModel):
    step: int = Field(ge=0)
    total_steps: int = Field(gt=0)
    time_of_day: str
    menu: list[MenuItem]
    staff: list[StaffMember]
    inventory: list[InventoryItem]
    active_orders: int = Field(ge=0)
    completed_orders: int = Field(ge=0)
    failed_orders: int = Field(ge=0)
    revenue: float = Field(ge=0)
    costs: float = Field(ge=0)
    customer_rating: float = Field(ge=1.0, le=5.0)
    demand_level: float = Field(gt=0)
    pending_customers: int = Field(ge=0)

class ShiftResult(BaseModel):
    total_revenue: float
    total_costs: float
    profit: float
    average_rating: float = Field(ge=1.0, le=5.0)
    orders_served: int = Field(ge=0)
    orders_failed: int = Field(ge=0)
    customer_satisfaction: float = Field(ge=0, le=100)