"""
Restaurant Manager OpenEnv — Inference Script
==============================================

Runs an LLM agent against all restaurant management tasks.

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   LLM API endpoint (default: HuggingFace router)
    MODEL_NAME     Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       Your Hugging Face API token (required)

OPTIONAL:
    TASK           Single task to run (default: runs ALL tasks)
    VERBOSE        Show detailed output (default: false)

STDOUT FORMAT (Competition-compliant):
    [START] task=<task> env=restaurant-manager model=<model>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

from openai import OpenAI

from env.models import AgentAction, RestaurantState

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

TASK_NAME = os.getenv("TASK", "all")   # "all" runs all 3 tasks
BENCHMARK = "restaurant-manager"
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

ALL_TASKS = ["weekday_lunch", "weekend_rush", "crisis_shift"]
MAX_STEPS = 12
MAX_SCORE = 100.0
SUCCESS_THRESHOLD = 60.0


def _debug(msg: str) -> None:
    """Print debug info to stderr only — never pollute stdout."""
    print(msg, file=sys.stderr, flush=True)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    score_normalized = score / MAX_SCORE
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score_normalized:.2f} rewards={rewards_str}",
        flush=True,
    )


SYSTEM_PROMPT = """You are an expert restaurant shift manager AI for an Indian restaurant.
You manage a 12-step shift (30 minutes each). Each step you decide:

1. STAFFING: Call in (true) or send home (false) staff by name
2. MENU: Enable (true) or disable (false) menu items
3. PRICING: Set new selling prices
4. INVENTORY: Emergency reorder quantities (costs 1.5x normal)
5. PROMOTIONS: 15% discount → +30% customer demand

=== STRATEGY ===
- High demand (>=1.5x): Call in ALL chefs and servers. Failing orders kills rating.
- Low rating (<4.0): Activate highest-skill staff only. Quality > quantity.
- Crisis costs (doubled prices): Disable expensive mains (Butter Chicken, Paneer Tikka).
  Focus on Naan, Mango Lassi, Dal Tadka — high margin, low ingredient cost.
- Low inventory: Only serve items you can actually make. Disable rest.
- Health inspection (step 8): MUST have active dishwasher skill >= 0.6 (Arjun).
- Large party: Call in extra chefs immediately.

=== CAPACITY ===
Chef capacity: ~5 orders/step × skill_level
Server capacity: ~8 orders/step × skill_level
Demand 1.0 = ~10 orders, 2.0 = ~20 orders, 2.5 = ~25 orders

=== RESPOND ONLY WITH VALID JSON — NO EXPLANATION, NO MARKDOWN ===
{
  "staff_changes": {"Ravi": true, "Amit": false},
  "menu_changes": {"Butter Chicken": false, "Naan": true},
  "price_adjustments": {"Naan": 70},
  "reorder_inventory": {"chicken": 10},
  "promotion_active": false
}
Only include fields you want to change. Omit unchanged fields."""


def state_to_prompt(state: RestaurantState) -> str:
    lines = [
        f"STEP {state.step + 1}/{state.total_steps} | {state.time_of_day}",
        f"Demand: {state.demand_level:.1f}x | Rating: {state.customer_rating:.2f}/5.0",
        f"Revenue: ₹{state.revenue:.0f} | Costs: ₹{state.costs:.0f} | Profit: ₹{state.revenue - state.costs:.0f}",
        f"Orders — Served: {state.completed_orders} | Failed: {state.failed_orders}",
        "",
        "STAFF (name | role | skill | wage | status):",
    ]
    for s in state.staff:
        status = "ACTIVE" if s.is_active else "idle"
        lines.append(f"  {s.name} | {s.role} | skill={s.skill_level:.1f} | ₹{s.hourly_wage}/h | [{status}]")

    lines += ["", "MENU (name | price | cost | margin | available):"]
    for item in state.menu:
        status = "ON" if item.available else "OFF"
        margin = item.price - item.cost
        lines.append(f"  {item.name} | ₹{item.price} | cost ₹{item.cost} | margin ₹{margin:.0f} | [{status}]")

    lines += ["", "INVENTORY (name | qty | unit | reorder cost | status):"]
    for inv in state.inventory:
        warn = " ⚠️LOW" if inv.quantity < 5 else ""
        lines.append(f"  {inv.name} | {inv.quantity:.1f} {inv.unit} | reorder ₹{inv.reorder_cost_per_unit}/unit{warn}")

    lines.append("\nDecide now. JSON only:")
    return "\n".join(lines)


def parse_llm_response(response_text: str) -> AgentAction:
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    if "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    try:
        return AgentAction(**json.loads(text))
    except Exception:
        return AgentAction()


def apply_safety_rules(state: RestaurantState, action: AgentAction) -> AgentAction:
    """
    Deterministic safety net that prevents obviously bad decisions.
    LLM decides strategy; these rules prevent catastrophic mistakes.
    """
    merged = action.model_copy(deep=True)

    # Rule 1: High demand → always have enough staff
    if state.demand_level >= 1.5:
        chefs = sorted([s for s in state.staff if s.role == "chef"], key=lambda x: -x.skill_level)
        servers = sorted([s for s in state.staff if s.role == "server"], key=lambda x: -x.skill_level)
        needed_chefs = max(2, int(state.demand_level * 1.5))
        needed_servers = max(2, int(state.demand_level))
        for chef in chefs[:needed_chefs]:
            if not chef.is_active:
                merged.staff_changes[chef.name] = True
        for server in servers[:needed_servers]:
            if not server.is_active:
                merged.staff_changes[server.name] = True

    # Rule 2: Always keep at least one dishwasher (health inspection safety)
    dishwashers = [s for s in state.staff if s.role == "dishwasher"]
    any_active_dw = any(
        (s.is_active and merged.staff_changes.get(s.name, True)) or
        (not s.is_active and merged.staff_changes.get(s.name) is True)
        for s in dishwashers
    )
    if not any_active_dw:
        best_dw = max(dishwashers, key=lambda x: x.skill_level)
        merged.staff_changes[best_dw.name] = True

    # Rule 3: Pre-inspection step (step 7 = before step 8 inspection) — ensure Arjun is active
    if state.step >= 6:
        arjun = next((s for s in state.staff if s.name == "Arjun"), None)
        if arjun and not arjun.is_active:
            merged.staff_changes["Arjun"] = True

    # Rule 4: Disable menu items we can't make (prevent failed orders)
    inv_lookup = {inv.name: inv.quantity for inv in state.inventory}
    for item in state.menu:
        can_make = all(
            inv_lookup.get(ing, 0) >= qty
            for ing, qty in item.ingredients.items()
        )
        if not can_make and item.available and item.name not in merged.menu_changes:
            merged.menu_changes[item.name] = False

    return merged


def run_episode(task_id: str, client: OpenAI) -> tuple[float, bool]:
    """Run a single task episode. Returns (final_score_0_to_100, success)."""
    from env.environment import RestaurantEnv
    from env.graders import grade

    last_error: list[Optional[str]] = [None]

    def policy(state: RestaurantState) -> AgentAction:
        last_error[0] = None
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": state_to_prompt(state)},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            llm_action = parse_llm_response(response.choices[0].message.content or "")
        except Exception as e:
            last_error[0] = str(e)[:200]
            if VERBOSE:
                _debug(f"  [LLM ERROR] {e}")
            llm_action = AgentAction()

        return apply_safety_rules(state, llm_action)

    env = RestaurantEnv()
    state = env.reset(task_id)

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: list[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False
    done = False

    try:
        step = 0
        while not done and step < MAX_STEPS:
            step += 1
            action = policy(state)
            state, reward, done, info = env.step(action)

            action_json = json.dumps({
                "staff_changes": action.staff_changes,
                "menu_changes": action.menu_changes,
                "price_adjustments": action.price_adjustments,
                "reorder_inventory": action.reorder_inventory,
                "promotion_active": action.promotion_active,
            }, separators=(",", ":"))

            log_step(step=step, action=action_json, reward=reward, done=done, error=last_error[0])
            rewards.append(reward)
            steps_taken = step

            if VERBOSE:
                _debug(
                    f"  Step {step:2d} | {state.time_of_day} | "
                    f"demand={state.demand_level:.1f} | "
                    f"served={info['orders_served']} failed={info['orders_failed']} | "
                    f"profit=₹{info['step_profit']:.0f} | rating={state.customer_rating:.2f} | reward={reward:.3f}"
                )

        result = env.get_result()
        grade_report = grade(task_id, result)
        final_score = grade_report.get("final_score", 0.0)
        success = final_score >= SUCCESS_THRESHOLD

        if VERBOSE:
            _debug(f"\n  ── {task_id.upper()} RESULT ──")
            _debug(f"  Revenue: ₹{result.total_revenue:,.0f} | Costs: ₹{result.total_costs:,.0f} | Profit: ₹{result.profit:,.0f}")
            _debug(f"  Rating: {result.average_rating:.2f} | Satisfaction: {result.customer_satisfaction:.1f}/100")
            _debug(f"  Served: {result.orders_served} | Failed: {result.orders_failed}")
            _debug(f"  FINAL SCORE: {final_score:.1f}/100 | {'✅ PASS' if success else '❌ FAIL'}")
            for pillar, pscore in grade_report.get("pillar_scores", {}).items():
                _debug(f"    {pillar}: {pscore:.1f}")

    except Exception as e:
        last_error[0] = str(e)[:200]
        if VERBOSE:
            _debug(f"  [ERROR] Episode failed: {e}")
            import traceback
            traceback.print_exc()
        success = False

    log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)
    return final_score, success


def main() -> None:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is required.", file=sys.stderr, flush=True)
        sys.exit(1)

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    if VERBOSE:
        _debug(f"[CONFIG] API: {API_BASE_URL}")
        _debug(f"[CONFIG] Model: {MODEL_NAME}")
        _debug(f"[CONFIG] Task: {TASK_NAME}")

    tasks_to_run = ALL_TASKS if TASK_NAME == "all" else [TASK_NAME]

    all_scores: list[float] = []
    all_success: list[bool] = []

    for task_id in tasks_to_run:
        score, success = run_episode(task_id, client)
        all_scores.append(score)
        all_success.append(success)

    if VERBOSE and len(tasks_to_run) > 1:
        _debug(f"\n{'='*60}")
        _debug(f"OVERALL RESULTS")
        for task_id, score, success in zip(tasks_to_run, all_scores, all_success):
            _debug(f"  {task_id}: {score:.1f}/100 {'✅' if success else '❌'}")
        _debug(f"  Average: {sum(all_scores) / len(all_scores):.1f}/100")
        _debug(f"{'='*60}")


if __name__ == "__main__":
    main()