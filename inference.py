"""
Uses OpenAI API to make management decisions based on restaurant state.

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL    LLM API endpoint (default: OpenAI)
    MODEL_NAME      Model identifier (default: gpt-4o-mini)
    OPENAI_API_KEY  API authentication key

OPTIONAL:
    TASK            Task to run (default: weekday_lunch)
    VERBOSE         Show detailed output (default: false)

STDOUT FORMAT (Competition-compliant):
    [START] task=<task> env=restaurant-manager model=<model>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>

Usage:
    python inference_competition.py                # Uses environment variables
    TASK=weekday_lunch python inference_competition.py # Set task via env var
    MODEL_NAME=gpt-4o python inference_competition.py  # Override model
    VERBOSE=true python inference_competition.py       # Show detailed output
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

from openai import OpenAI

from env.models import AgentAction, RestaurantState


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION (from environment or defaults)
# ═══════════════════════════════════════════════════════════════════════

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY")

TASK_NAME = os.getenv("TASK", "weekday_lunch")
BENCHMARK = "restaurant-manager"
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

MAX_STEPS = 12
MAX_SCORE = 100.0  # Grader returns 0-100
SUCCESS_THRESHOLD = 65.0  # Need 65/100 to pass

# ═══════════════════════════════════════════════════════════════════════
# LOGGING FUNCTIONS (Competition format)
# ═══════════════════════════════════════════════════════════════════════


def log_start(task: str, env: str, model: str) -> None:
    """Log episode start in competition format."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    """Log step completion in competition format."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    """Log episode end in competition format."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    score_normalized = score / MAX_SCORE  # Normalize to [0, 1]
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score_normalized:.3f} rewards={rewards_str}",
        flush=True,
    )


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
    text = response_text.strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    # Extract JSON object
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]

    try:
        data = json.loads(text)
        return AgentAction(**data)
    except (json.JSONDecodeError, Exception) as e:
        if VERBOSE:
            print(f"  [WARNING] Failed to parse LLM response: {e}", flush=True)
        return AgentAction()


# ═══════════════════════════════════════════════════════════════════════
# LLM POLICY
# ═══════════════════════════════════════════════════════════════════════


def make_llm_policy(model: str, verbose: bool = False):
    """
    Create an LLM-powered policy function.

    Args:
        model: Model identifier to use
        verbose: if True, print LLM responses

    Returns:
        A policy function: RestaurantState -> AgentAction
    """
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

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
                print(f"  [LLM] {response_text[:150]}...", flush=True)

            return parse_llm_response(response_text)

        except Exception as e:
            if verbose:
                print(f"  [ERROR] LLM call failed: {e}", flush=True)
            return AgentAction()

    return policy


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    """Run the agent following competition submission format."""
    
    # Validate API key
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set.", flush=True)
        sys.exit(1)

    # Log episode start
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    steps_taken = 0
    rewards: list[float] = []
    final_score = 0.0
    success = False

    try:
        # Import here to avoid circular imports
        from env.environment import RestaurantEnv
        from env.graders import grade
        
        # Create policy and environment
        policy = make_llm_policy(model=MODEL_NAME, verbose=VERBOSE)
        env = RestaurantEnv()
        state = env.reset(TASK_NAME)

        # Run step-by-step
        done = False
        step = 0
        
        while not done and step < MAX_STEPS:
            step += 1
            
            # Get agent action
            action = policy(state)
            
            # Execute in environment
            state, reward, done, info = env.step(action)
            
            # Format action as JSON string
            action_json = json.dumps({
                "staff_changes": action.staff_changes,
                "menu_changes": action.menu_changes,
                "price_adjustments": action.price_adjustments,
                "reorder_inventory": action.reorder_inventory,
                "promotion_active": action.promotion_active,
            })
            
            # Log step in competition format
            log_step(
                step=step,
                action=action_json,
                reward=reward,
                done=done,
                error=None,
            )
            
            rewards.append(reward)
            steps_taken = step

        # Get final results
        result = env.get_result()
        grade_report = grade(TASK_NAME, result)
        final_score = grade_report.get("final_score", 0.0)
        
        # Determine success
        success = final_score >= SUCCESS_THRESHOLD

        if VERBOSE:
            print(f"\n{'='*60}", flush=True)
            print(f"Task: {TASK_NAME}", flush=True)
            print(f"Final Score: {final_score:.1f}/100", flush=True)
            for pillar, score in grade_report.get("pillar_scores", {}).items():
                print(f"  {pillar}: {score:.1f}/100", flush=True)
            print(f"{'='*60}\n", flush=True)

    except Exception as e:
        if VERBOSE:
            print(f"[ERROR] Episode failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        success = False
        steps_taken = 0
        rewards = []
        final_score = 0.0

    finally:
        # Always log end (even on error)
        log_end(
            success=success,
            steps=steps_taken,
            score=final_score,
            rewards=rewards,
        )


if __name__ == "__main__":
    main()
