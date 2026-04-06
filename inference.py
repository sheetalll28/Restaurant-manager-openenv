"""
AI agent for the restaurant management environment.

Uses OpenAI's API to make management decisions based on the current
restaurant state. This is the main hackathon deliverable.

Usage:
    python inference.py                          # runs all 3 tasks
    python inference.py --task weekday_lunch      # runs one task
    python inference.py --task weekday_lunch --verbose  # detailed output

Requires:
    OPENAI_API_KEY environment variable to be set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from openai import OpenAI

from env.models import AgentAction, RestaurantState
from env.simulation import run_episode


# ═══════════════════════════════════════════════════════════════════════
# PROMPT CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert restaurant shift manager AI. You manage a restaurant during a single shift (12 steps, 30 minutes each).

=== YOUR 5 DECISION LEVERS (each step) ===
1. STAFFING: Call in or send home staff (chefs, servers, dishwashers)
2. MENU: Enable or disable menu items
3. PRICING: Adjust item prices (cost + margin)
4. INVENTORY: Emergency-reorder ingredients (1.5x normal cost)
5. PROMOTIONS: Run a 15% discount promotion to boost demand by 30%

=== YOUR 4 GOALS (priority order) ===
Priority 1: SERVE CUSTOMERS → minimize failed orders (need enough chefs & servers)
Priority 2: MAINTAIN RATING → keep above 4.0 (better staff skill = higher ratings)
Priority 3: BE PROFITABLE → revenue > costs (don't overspend)
Priority 4: SPECIAL EVENTS → handle large parties & health inspections

=== KEY RULES ===
- Chef capacity: ~5 orders/step per chef (scaled by skill_level)
- Server capacity: ~8 orders/step per server (scaled by skill_level)
- Demand scaling: demand=1.0 → ~10 orders, demand=2.0 → ~20 orders
- Promotion effect: +30% demand, -15% price
- Staff cost: hourly_wage × 0.5 per 30-minute step
- Health inspections: need active dishwasher with skill ≥ 0.6

=== RESPONSE FORMAT (REQUIRED) ===
RESPOND ONLY with a valid JSON object. No explanation. No markdown code blocks.
Use this exact schema (omit fields you don't change):

{
  "staff_changes": {"Alice": true, "Bob": false},
  "menu_changes": {"Burger": true, "Pasta": false},
  "price_adjustments": {"Burger": 12.99, "Pizza": 14.50},
  "reorder_inventory": {"Beef": 50, "Tomato": 100},
  "promotion_active": true
}

FIELD GUIDE:
- staff_changes: true=call in, false=send home
- menu_changes: true=enable, false=disable
- price_adjustments: new selling price (must be > cost)
- reorder_inventory: quantity to order (emergency reorder)
- promotion_active: boolean (true/false only)

=== EXAMPLE SCENARIOS ===

Example 1 (High demand, staff shortage):
{
  "staff_changes": {"Chef_Maria": true, "Server_John": true},
  "promotion_active": false
}

Example 2 (Slow period, over-staffed):
{
  "staff_changes": {"Chef_Bob": false},
  "menu_changes": {"ExpensiveItem": false},
  "promotion_active": true
}

Example 3 (Profit focus, low inventory):
{
  "price_adjustments": {"Burger": 13.50, "Pizza": 15.00},
  "reorder_inventory": {"Beef": 30, "Tomato": 50}
}

=== CRITICAL INSTRUCTIONS ===
1. Always output VALID JSON only (no explanations, no code blocks, no extra text)
2. Only include fields you want to change (omit the rest)
3. Use exact staff/menu/ingredient names from the state
4. Ensure new prices are ABOVE cost
5. Make BOLD decisions when demand is high or rating is low
6. Be CONSERVATIVE with spending during profitable periods
"""


def state_to_prompt(state: RestaurantState) -> str:
    """Convert a RestaurantState to a human-readable prompt for the LLM."""
    lines = []

    lines.append(f"=== STEP {state.step + 1}/{state.total_steps} | Time: {state.time_of_day} ===")
    lines.append(f"Demand Level: {state.demand_level}x | Rating: {state.customer_rating}/5.0")
    lines.append(f"Revenue so far: {state.revenue:.0f} | Costs so far: {state.costs:.0f} | "
                 f"Profit: {state.revenue - state.costs:.0f}")
    lines.append(f"Orders served: {state.completed_orders} | Failed: {state.failed_orders}")
    lines.append("")

    # Staff
    lines.append("STAFF:")
    for s in state.staff:
        status = "ACTIVE" if s.is_active else "available"
        lines.append(f"  {s.name} | {s.role} | skill={s.skill_level} | wage={s.hourly_wage}/hr | {status}")
    lines.append("")

    # Menu
    lines.append("MENU:")
    for item in state.menu:
        status = "ON" if item.available else "OFF"
        margin = item.price - item.cost
        lines.append(f"  {item.name} | price={item.price} cost={item.cost} margin={margin:.0f} | "
                     f"prep={item.prep_time_minutes}min | [{status}]")
    lines.append("")

    # Inventory
    lines.append("INVENTORY:")
    for inv in state.inventory:
        warning = " ⚠️ LOW" if inv.quantity < 5 else ""
        lines.append(f"  {inv.name}: {inv.quantity:.1f} {inv.unit} | "
                     f"cost={inv.cost_per_unit}/{inv.unit} | "
                     f"reorder={inv.reorder_cost_per_unit}/{inv.unit}{warning}")
    lines.append("")

    lines.append("What actions should I take this step? Respond with JSON only.")
    return "\n".join(lines)


def parse_llm_response(response_text: str) -> AgentAction:
    """Parse the LLM's JSON response into an AgentAction."""
    # Clean and extract JSON from the response
    text = response_text.strip()

    # Handle markdown code blocks (```json...``` or ```...```)
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    # Try to find JSON object (handles text before/after JSON)
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]

    try:
        data = json.loads(text)
        return AgentAction(**data)
    except (json.JSONDecodeError, Exception) as e:
        # If parsing fails, return a do-nothing action
        print(f"  [WARNING] Failed to parse LLM response: {e}")
        print(f"  [DEBUG] Extracted text: {text[:200]}")
        return AgentAction()


# ═══════════════════════════════════════════════════════════════════════
# LLM POLICY
# ═══════════════════════════════════════════════════════════════════════


def make_llm_policy(model: str = "gpt-4o-mini", verbose: bool = False):
    """
    Create an LLM-powered policy function.

    Args:
        model: OpenAI model name (default: gpt-4o-mini)
        verbose: if True, print LLM responses

    Returns:
        A policy function: RestaurantState -> AgentAction
    """
    # Configure OpenAI (reads OPENAI_API_KEY from environment)
    client = OpenAI()

    def policy(state: RestaurantState) -> AgentAction:
        prompt = state_to_prompt(state)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=700,
            )

            response_text = response.choices[0].message.content or ""

            if verbose:
                print(f"  [LLM] {response_text[:150]}...")

            return parse_llm_response(response_text)

        except Exception as e:
            print(f"  [ERROR] LLM call failed: {e}")
            return AgentAction()  # fallback: do nothing

    return policy


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Run AI agent on restaurant tasks")
    parser.add_argument(
        "--task",
        choices=["weekday_lunch", "weekend_rush", "crisis_shift", "all"],
        default="all",
        help="Which task to run (default: all)",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Set it with: $env:OPENAI_API_KEY = 'your-key-here'")
        sys.exit(1)

    tasks = (
        ["weekday_lunch", "weekend_rush", "crisis_shift"]
        if args.task == "all"
        else [args.task]
    )

    policy = make_llm_policy(model=args.model, verbose=args.verbose)

    results = {}
    for task_id in tasks:
        print(f"\n{'='*60}")
        print(f"  Running: {task_id} (model: {args.model})")
        print(f"{'='*60}")

        result, grade_report = run_episode(task_id, policy, verbose=args.verbose)
        results[task_id] = grade_report

        print(f"\n  Result: profit={result.profit:.0f} | rating={result.average_rating} | "
              f"served={result.orders_served} | failed={result.orders_failed}")
        print(f"  SCORE: {grade_report['final_score']}/100")
        for pillar, score in grade_report["pillar_scores"].items():
            print(f"    {pillar}: {score:.1f}/100")

    # Summary
    if len(tasks) > 1:
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        total = 0
        for task_id, report in results.items():
            score = report["final_score"]
            total += score
            print(f"  {task_id:<20s}: {score:.1f}/100")
        avg = total / len(results)
        print(f"  {'AVERAGE':<20s}: {avg:.1f}/100")


if __name__ == "__main__":
    main()
