"""
Tests for env/models.py — verify validation and defaults.

Run with:  pytest tests/test_models.py -v
"""

import pytest
from pydantic import ValidationError

from env.models import (
    AgentAction,
    InventoryItem,
    MenuItem,
    ShiftResult,
    StaffMember,
)


# ── MenuItem tests ─────────────────────────────────────────────────────


class TestMenuItem:
    """Tests for the MenuItem model."""

    def test_valid_menu_item(self):
        """A fully valid menu item should be created without errors."""
        item = MenuItem(
            name="Butter Chicken",
            price=350,
            cost=120,
            prep_time_minutes=20,
            category="main",
            ingredients={"chicken": 0.5, "spices": 0.1},
        )
        assert item.name == "Butter Chicken"
        assert item.price == 350
        assert item.available is True  # default

    def test_negative_price_rejected(self):
        """Price must be > 0."""
        with pytest.raises(ValidationError):
            MenuItem(
                name="Bad",
                price=-10,
                cost=5,
                prep_time_minutes=10,
                category="main",
                ingredients={},
            )

    def test_invalid_category_rejected(self):
        """Category must be one of: appetizer, main, dessert, drink."""
        with pytest.raises(ValidationError):
            MenuItem(
                name="Bad",
                price=100,
                cost=50,
                prep_time_minutes=10,
                category="sushi",  # not a valid category
                ingredients={},
            )

    def test_available_defaults_to_true(self):
        """If we don't specify 'available', it should default to True."""
        item = MenuItem(
            name="Naan",
            price=60,
            cost=15,
            prep_time_minutes=5,
            category="appetizer",
            ingredients={"flour": 0.2},
        )
        assert item.available is True


# ── StaffMember tests ─────────────────────────────────────────────────


class TestStaffMember:
    """Tests for the StaffMember model."""

    def test_valid_staff(self):
        chef = StaffMember(
            name="Ravi", role="chef", skill_level=0.9, hourly_wage=200
        )
        assert chef.role == "chef"
        assert chef.is_active is False  # default

    def test_invalid_role_rejected(self):
        """Role must be chef, server, or dishwasher."""
        with pytest.raises(ValidationError):
            StaffMember(
                name="X", role="pilot", skill_level=0.5, hourly_wage=100
            )

    def test_skill_level_out_of_range_rejected(self):
        """Skill level must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            StaffMember(
                name="X", role="chef", skill_level=1.5, hourly_wage=100
            )


# ── InventoryItem tests ───────────────────────────────────────────────


class TestInventoryItem:
    """Tests for the InventoryItem model."""

    def test_valid_inventory(self):
        inv = InventoryItem(
            name="Rice",
            quantity=50,
            unit="kg",
            cost_per_unit=60,
            reorder_cost_per_unit=90,
        )
        assert inv.quantity == 50

    def test_negative_quantity_rejected(self):
        """Quantity cannot be negative."""
        with pytest.raises(ValidationError):
            InventoryItem(
                name="Rice",
                quantity=-5,
                unit="kg",
                cost_per_unit=60,
                reorder_cost_per_unit=90,
            )


# ── AgentAction tests ─────────────────────────────────────────────────


class TestAgentAction:
    """Tests for the AgentAction model."""

    def test_empty_action(self):
        """An action with all defaults = do nothing. Should be valid."""
        action = AgentAction()
        assert action.staff_changes == {}
        assert action.menu_changes == {}
        assert action.price_adjustments == {}
        assert action.reorder_inventory == {}
        assert action.promotion_active is False

    def test_action_with_changes(self):
        """An action with some decisions filled in."""
        action = AgentAction(
            staff_changes={"Ravi": True, "Priya": False},
            price_adjustments={"Butter Chicken": 400},
            promotion_active=True,
        )
        assert action.staff_changes["Ravi"] is True
        assert action.promotion_active is True


# ── ShiftResult tests ─────────────────────────────────────────────────


class TestShiftResult:
    """Tests for the ShiftResult model."""

    def test_valid_result(self):
        result = ShiftResult(
            total_revenue=50000,
            total_costs=30000,
            profit=20000,
            average_rating=4.2,
            orders_served=120,
            orders_failed=5,
            customer_satisfaction=85.0,
        )
        assert result.profit == 20000

    def test_rating_out_of_range_rejected(self):
        """Rating must be between 1.0 and 5.0."""
        with pytest.raises(ValidationError):
            ShiftResult(
                total_revenue=1000,
                total_costs=500,
                profit=500,
                average_rating=6.0,  # too high
                orders_served=10,
                orders_failed=0,
                customer_satisfaction=90,
            )
