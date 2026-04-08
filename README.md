---
title: Restaurant Manager OpenEnv
emoji: 🍽️
colorFrom: orange
colorTo: red
sdk: docker
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - restaurant-management
  - real-world-rl
  - indian-market
license: mit
---

# Restaurant Manager — OpenEnv

An RL environment where an AI agent manages a restaurant shift. Not a toy. The agent has to balance staff wages against customer demand, keep ratings up, manage ingredients that run out mid-service, survive health inspections, and stay profitable under supply chain pressure.

Built for the OpenEnv RL Challenge.

---

## What the agent actually does

Every 30 minutes (one step), the agent looks at the full restaurant state — who's working, what's on the menu, what's in the fridge, how many customers are coming in — and makes decisions:

- **Staffing** — call in or send home any of 8 staff. Chefs handle order volume, servers deliver, dishwashers are mandatory for health inspections.
- **Menu** — disable dishes you can't make (out of stock) or don't have capacity to prep. Leaving unavailable items "on" causes failed orders and kills your rating.
- **Pricing** — adjust selling prices in real time. Useful for margins during low-demand periods.
- **Inventory reorder** — emergency mid-shift reorders at 1.5x normal cost. Sometimes worth it, sometimes not.
- **Promotions** — 15% discount that pulls 30% more customers. Great at low-demand; dangerous at peak when you can't handle more volume.

The simulation tracks ingredient consumption, staff capacity (skills matter — a 0.9 chef handles ~4.5 orders/step, a 0.5 chef only 2.5), rolling customer ratings, and cumulative profit. Fail too many orders and your rating tanks. Low rating = lower future demand. It compounds.

---

## Three tasks, genuinely different

**weekday_lunch** (easy) — Full inventory, good starting rating (4.3), predictable demand curve peaking at lunch. The agent mostly needs to scale staff to the 1.5x demand peak and avoid obvious mistakes. Good for learning the basics.

**weekend_rush** (medium) — The best chef is unavailable. Starting rating is already damaged at 3.5 (one bad review). Demand hits 2.5x. A large party of 15 walks in at step 4 with no warning. The agent needs to recover the rating while handling surge demand with second-tier staff. Rating improvement is weighted highest in scoring — profit matters less.

**crisis_shift** (hard) — Supply chain crisis doubled all ingredient costs. Inventory starts at 40% of normal. A health inspector arrives at step 8 (you need Arjun the dishwasher active with skill ≥ 0.6 or you get a rating penalty). A competitor opened nearby so demand is softer. Three things go wrong simultaneously and the agent has to prioritize correctly.

---

## Reward function

Per-step reward, not terminal-only:

```
reward = (0.4 × tanh(profit/5000)) + (0.3 × (rating−1)/4) + (0.3 × served/total)
reward *= (1 − failure_rate)^1.5
```

The failure penalty is exponential — 50% failed orders reduces your reward by 29%, 100% failure wipes it entirely. This means the agent learns quickly that failed orders are worse than understaffing.

Reward range is approximately −0.5 to +0.75 per step.

Final score (0–100) is calculated separately by a deterministic grader that uses task-specific profit targets, rating targets, and service rate targets with different pillar weights per task.

---

## Scoring breakdown

| Task | Profit | Rating | Service | Satisfaction |
|---|---|---|---|---|
| weekday_lunch | 30% | 25% | 25% | 20% |
| weekend_rush | 25% | **35%** | 25% | 15% |
| crisis_shift | **35%** | 25% | 15% | 25% |

The weighting shifts match the actual challenge. Weekend rush is won or lost on whether you recover the rating. Crisis shift is won or lost on whether you stay profitable under doubled costs.

---

## Baseline scores (do-nothing policy)

Just to have a floor for comparison:

| Task | Score | Notes |
|---|---|---|
| weekday_lunch | 81.2 / 100 | Default staff setup is decent, demand is mild |
| weekend_rush | 51.7 / 100 | High demand + low rating = lots of failed orders |
| crisis_shift | 60.5 / 100 | Low inventory self-limits damage |

A good LLM agent should beat these on all three tasks.

---

## Setup

```bash
git clone https://github.com/sheetalll28/Restaurant-manager-openenv
cd Restaurant-manager-openenv
pip install -r requirements.txt

# Start the server
uvicorn app:app --host 0.0.0.0 --port 7860

# Run inference against all tasks
HF_TOKEN=your_token python inference.py

# Run one specific task
HF_TOKEN=your_token TASK=crisis_shift VERBOSE=true python inference.py
```

### Docker

```bash
docker build -t restaurant-manager .
docker run -p 7860:7860 -e HF_TOKEN=your_token restaurant-manager
```

### API

```python
import httpx

# Start an episode
r = httpx.post("http://localhost:7860/reset", json={"task_id": "weekend_rush"})
state = r.json()["observation"]

# Take a step
action = {
    "staff_changes": {"Priya": True, "Sneha": True, "Arjun": True},
    "menu_changes": {"Butter Chicken": True},
    "promotion_active": False
}
r = httpx.post("http://localhost:7860/step", json=action)
print(r.json()["reward"], r.json()["done"])
```

### Validate

```bash
pip install openenv-core
openenv validate .
```

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | **yes** | — | Hugging Face API token |
| `API_BASE_URL` | no | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | no | `Qwen/Qwen2.5-72B-Instruct` | Model to use |
| `TASK` | no | `all` | Task to run (`all`, `weekday_lunch`, `weekend_rush`, `crisis_shift`) |
| `VERBOSE` | no | `false` | Print step-by-step debug info to stderr |

---

## Project structure

```
├── inference.py        ← competition submission script (must be in root)
├── app.py              ← FastAPI server (OpenEnv HTTP interface)
├── openenv.yaml        ← spec metadata
├── Dockerfile
├── requirements.txt
├── README.md
└── env/
    ├── environment.py  ← simulation logic (step/reset/state)
    ├── models.py       ← Pydantic types
    ├── graders.py      ← deterministic scoring
    ├── tasks.py        ← task configs
    └── runner.py       ← episode runner utility
```

---

## Staff and menu reference

**Staff:**
Ravi (chef, 0.9 skill, ₹250/hr) · Priya (chef, 0.7, ₹200) · Amit (chef, 0.5, ₹150) · Sneha (server, 0.85, ₹150) · Vikram (server, 0.6, ₹120) · Meera (server, 0.7, ₹130) · Arjun (dishwasher, 0.8, ₹100) · Kavita (dishwasher, 0.5, ₹90)

**Menu:**
Butter Chicken ₹350 · Paneer Tikka ₹280 · Dal Tadka ₹180 · Naan ₹60 · Gulab Jamun ₹120 · Mango Lassi ₹100

Prices and wages are calibrated against Indian restaurant market data (Swiggy/Zomato range).