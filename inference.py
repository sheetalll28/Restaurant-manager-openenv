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
from urllib.parse import urlparse

from openai import OpenAI

from env.models import AgentAction, RestaurantState


# ═══════════════════════════════════════════════════════════════════════
# PROVIDER DETECTION
# ═══════════════════════════════════════════════════════════════════════

def detect_provider(base_url: str) -> str:
    """Detect the LLM provider from the base_url."""
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if "localhost" in host or "127.0.0.1" in host:
        if "11434" in str(parsed.port):
            return "OLLAMA (local)"
        return f"LOCAL ({host}:{parsed.port})"
    elif "api.openai.com" in host:
        return "OPENAI (cloud)"
    else:
        return f"OTHER ({host})"

def _debug(msg: str) -> None:
    """Print debug info to stderr (never pollute stdout)."""
    print(msg, file=sys.stderr, flush=True)


def warn_if_model_mismatch(provider: str, model: str) -> None:
    """Print a warning if model name doesn't match the provider."""
    openai_models = {"gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o1-mini"}
    is_ollama = "OLLAMA" in provider or "LOCAL" in provider
    if is_ollama and model in openai_models:
        _debug(f"  ⚠️  WARNING: model='{model}' is an OpenAI name but provider is {provider}")
        _debug(f"  ⚠️  Ollama does NOT have {model}. Use the model you pulled (e.g. 'mistral').")
        _debug(f"  ⚠️  Run 'ollama list' to see available models.")


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION (from environment or defaults)
# ═══════════════════════════════════════════════════════════════════════

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")

TASK_NAME = os.getenv("TASK", "weekday_lunch")
BENCHMARK = "restaurant-manager"
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
DETECTED_PROVIDER = detect_provider(API_BASE_URL)

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
2. MENU: Enable or disable menu items (disable low-margin or hard-to-prepare items)
3. PRICING: Adjust item prices (cost + margin)
4. INVENTORY: Emergency-reorder ingredients (1.5x normal cost)
5. PROMOTIONS: Run a 15% discount promotion to boost demand by 30%

=== CRITICAL STRATEGY RULES ===
** RATING is currency — low ratings hurt more than low profit **
- HIGH demand shift? Call in MORE staff to handle surge → serve more orders → better rating
- LOW rating? Focus on quality and speed → use high-skill staff → reduce failed orders
- COMPETITOR nearby? Don't race on price, compete on SERVICE QUALITY (fast, accurate)
- Cost spike? Disable expensive items, focus on high-margin items, reorder strategically

=== DECISION LOGIC ===
WHEN demand is high (demand >= 1.5):
  → Call in extra chefs and servers (MUST serve customers to keep rating)
  → Keep all profitable items enabled
  → Don't worry about small costs — failed orders cost rating points (worse)

WHEN rating is low (< 4.0):
  → Call in your BEST staff (highest skill)
  → Disable complex items (prep time > 15 min) to reduce failures
  → Don't promote — focus on quality, not volume

WHEN costs are high (doubled inventory prices):
  → Disable expensive dishes
  → Focus on efficient, high-margin items (Naan, Mango Lassi)
  → Strategic reorder: only order what you KNOW you'll sell
  → Better to miss an order than reorder at 2x cost

WHEN inventory is low:
  → Only enable items you have enough for
  → Don't take risks — avoid items requiring rare ingredients
  → Reorder ONLY essentials at premium cost

HEALTH INSPECTION (step 8):
  → Must have active dishwasher with skill >= 0.6
  → Quality matters — keep best chefs and servers active

LARGE PARTY (announced):
  → Ensure you have EXTRA chefs and servers available
  → High-skill staff can handle surges
  → Worth the extra payroll cost — failed orders from party hurt rating badly

=== CAPACITY MATH ===
- Chef capacity: ~5 orders/step per chef (skill_level 0.5=2.5 orders, skill_level 0.9=4.5 orders)
- Server capacity: ~8 orders/step per server (scaled by skill)
- Demand: 1.0 = ~10 orders, 1.5 = ~15 orders, 2.5 = ~25 orders
- Promotion: +30% demand, -15% margin
- Failed order cost: -rating penalty + no revenue (severe)

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
- reorder_inventory: quantity to order (emergency reorder at 1.5x cost)
- promotion_active: boolean (true/false only)
"""


def state_to_prompt(state: RestaurantState) -> str:
    """Convert a RestaurantState to a human-readable prompt for the LLM.
    
    Optimized to avoid redundant formatting on static data.
    """
    lines = []

    # Dynamic header (changes every step)
    lines.append(f"=== STEP {state.step + 1}/{state.total_steps} | Time: {state.time_of_day} ===")
    lines.append(f"Demand: {state.demand_level}x | Rating: {state.customer_rating:.1f}/5.0")
    profit = state.revenue - state.costs
    lines.append(f"Revenue: ${state.revenue:.0f} | Costs: ${state.costs:.0f} | Profit: ${profit:.0f}")
    lines.append(f"Served: {state.completed_orders} | Failed: {state.failed_orders}")
    lines.append("")

    # Staff (brief format)
    lines.append("STAFF:")
    for s in state.staff:
        status = "ON" if s.is_active else "off"
        lines.append(f"  {s.name} ({s.role}) skill={s.skill_level} wage=${s.hourly_wage}/h [{status}]")
    lines.append("")

    # Menu (compact format)
    lines.append("MENU:")
    for item in state.menu:
        status = "ON" if item.available else "OFF"
        margin = item.price - item.cost
        lines.append(f"  {item.name}: ${item.price} (cost ${item.cost}, margin ${margin:.0f}) [{status}]")
    lines.append("")

    # Inventory (compact format, only show low stock warnings)
    lines.append("INVENTORY:")
    for inv in state.inventory:
        status = "⚠️ LOW" if inv.quantity < 5 else "OK"
        lines.append(f"  {inv.name}: {inv.quantity:.0f}/{inv.unit} [{status}]")
    lines.append("")

    lines.append("Action? Respond JSON only.")
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
    Create a hybrid LLM + rule-based policy function.
    
    The LLM makes decisions, but we apply automatic rules for tough scenarios:
    - High demand surge? Auto call in extra staff
    - Low rating? Auto activate best staff
    - Doubled costs? Auto disable expensive items
    - Low inventory? Auto carefully manage reorders

    Args:
        model: Model identifier to use
        verbose: if True, print LLM responses

    Returns:
        A policy function: RestaurantState -> AgentAction
    """
    # ── Debug: Print provider info to stderr ──
    _debug(f"  [DEBUG] Provider  : {DETECTED_PROVIDER}")
    _debug(f"  [DEBUG] base_url  : {API_BASE_URL}")
    _debug(f"  [DEBUG] model     : {model}")
    warn_if_model_mismatch(DETECTED_PROVIDER, model)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    def apply_hybrid_rules(state: RestaurantState, llm_action: AgentAction) -> AgentAction:
        """
        Apply strategic rules to enhance the LLM decision.
        Rules are only applied when conditions are difficult.
        """
        merged = llm_action.model_copy(deep=True)
        
        # ═══════════════════════════════════════════════════════════════════
        # RULE 1: High demand surge → ensure enough staff
        # ═══════════════════════════════════════════════════════════════════
        if state.demand_level >= 1.5:
            # Count current active chefs and servers
            active_chefs = sum(1 for s in state.staff if s.is_active and s.role == "chef")
            active_servers = sum(1 for s in state.staff if s.is_active and s.role == "server")
            
            # For high demand, we need at least 3 chefs and 2 servers
            target_chefs = max(3, int(state.demand_level * 2))
            target_servers = max(2, int(state.demand_level * 1.5))
            
            # Auto call in best available staff if we're under target
            if active_chefs < target_chefs or active_servers < target_servers:
                # Get all staff sorted by skill
                chefs = [s for s in state.staff if s.role == "chef"]
                servers = [s for s in state.staff if s.role == "server"]
                
                chefs.sort(key=lambda x: x.skill_level, reverse=True)
                servers.sort(key=lambda x: x.skill_level, reverse=True)
                
                # Call in best chefs
                for chef in chefs[:target_chefs]:
                    if not chef.is_active:
                        merged.staff_changes[chef.name] = True
                
                # Call in best servers
                for server in servers[:target_servers]:
                    if not server.is_active:
                        merged.staff_changes[server.name] = True
                
                if verbose:
                    _debug(f"  [RULE] High demand ({state.demand_level}x) → calling in extra staff")
        
        # ═══════════════════════════════════════════════════════════════════
        # RULE 2: Low rating → activate best staff for quality
        # ═══════════════════════════════════════════════════════════════════
        if state.customer_rating < 4.0:
            # Get best staff by skill
            chefs = sorted(
                [s for s in state.staff if s.role == "chef"],
                key=lambda x: x.skill_level, reverse=True
            )
            servers = sorted(
                [s for s in state.staff if s.role == "server"],
                key=lambda x: x.skill_level, reverse=True
            )
            
            # Ensure at least top 2 chefs and top 2 servers are active
            for chef in chefs[:2]:
                if not chef.is_active:
                    merged.staff_changes[chef.name] = True
            
            for server in servers[:2]:
                if not server.is_active:
                    merged.staff_changes[server.name] = True
            
            if verbose:
                _debug(f"  [RULE] Low rating ({state.customer_rating:.1f}) → activating best staff")
        
        # ═══════════════════════════════════════════════════════════════════
        # RULE 3: Doubled ingredient costs → disable expensive items
        # ═══════════════════════════════════════════════════════════════════
        # Detect doubled costs by checking if any item has doubled cost_per_unit
        avg_cost = sum(inv.cost_per_unit for inv in state.inventory) / len(state.inventory)
        is_crisis_cost = avg_cost > 150  # Normal avg is ~80, crisis is ~160+
        
        if is_crisis_cost:
            # Disable expensive main courses, keep fast/simple items
            for item in state.menu:
                if item.category == "main" and item.cost > 100:
                    # Expensive main → disable it
                    merged.menu_changes[item.name] = False
                elif item.category in ("appetizer", "dessert", "drink"):
                    # Keep simple items enabled
                    merged.menu_changes[item.name] = True
            
            if verbose:
                _debug(f"  [RULE] Crisis costs detected → disabling expensive mains")
        
        # ═══════════════════════════════════════════════════════════════════
        # RULE 4: Low inventory → don't reorder unless critical
        # ═══════════════════════════════════════════════════════════════════
        low_inv_count = sum(1 for inv in state.inventory if inv.quantity < 5)
        if low_inv_count > 3:  # Multiple ingredients running low
            # Clear reorder queue — reordering at 1.5x cost in crisis is expensive
            merged.reorder_inventory = {}
            
            if verbose:
                _debug(f"  [RULE] Low inventory → skipping reorders to conserve costs")
        
        return merged

    # Track last error for competition logging
    last_error: list[str | None] = [None]

    def policy(state: RestaurantState) -> AgentAction:
        prompt = state_to_prompt(state)
        last_error[0] = None  # reset each step

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=400,
            )

            response_text = response.choices[0].message.content or ""
            response_model = getattr(response, "model", "unknown")

            if verbose:
                _debug(f"  [LLM] Response model: {response_model}")
                _debug(f"  [LLM] {response_text[:100]}...")
                if response_model != model:
                    _debug(f"  [LLM] ⚠️  Model mismatch! Requested='{model}', Got='{response_model}'")

            llm_action = parse_llm_response(response_text)

        except Exception as e:
            last_error[0] = str(e)[:200]
            if verbose:
                _debug(f"  [ERROR] LLM call failed: {e}")
            llm_action = AgentAction()
        
        # Apply hybrid rules to enhance LLM decision
        final_action = apply_hybrid_rules(state, llm_action)
        
        return final_action

    def get_last_error() -> str | None:
        return last_error[0]

    return policy, get_last_error


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
        policy, get_last_error = make_llm_policy(model=MODEL_NAME, verbose=VERBOSE)
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
                error=get_last_error(),
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
            _debug(f"\n{'='*60}")
            _debug(f"Task: {TASK_NAME}")
            _debug(f"Final Score: {final_score:.1f}/100")
            for pillar, score in grade_report.get("pillar_scores", {}).items():
                _debug(f"  {pillar}: {score:.1f}/100")
            _debug(f"{'='*60}\n")

    except Exception as e:
        if VERBOSE:
            _debug(f"[ERROR] Episode failed: {e}")
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
