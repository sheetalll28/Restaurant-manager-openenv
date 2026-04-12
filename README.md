---
title: Restaurant Manager OpenEnv
emoji: 🍽️
colorFrom: red
colorTo: orange
sdk: docker
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - restaurant-management
  - real-world-rl
  - simulation
license: mit
---

# Restaurant Manager OpenEnv

Restaurant Manager OpenEnv is a restaurant operations benchmark where an agent runs a 12-step shift inside an Indian restaurant setting. The agent has to balance staffing, menu availability, pricing, promotions, and emergency reorders while protecting service quality, ratings, and profitability.

This repository now includes:
- 8 distinct scenarios
- dense step rewards with stronger penalties for failures and inefficiency
- deterministic final grading
- a browser dashboard at `/play`
- OpenEnv-compatible packaging and validation

## What the agent controls

At every step, the agent receives the full restaurant state and chooses:

- `staff_changes`: call in or send home named staff members
- `menu_changes`: enable or disable menu items
- `price_adjustments`: raise or lower dish prices within guarded limits
- `reorder_inventory`: buy emergency stock at premium reorder prices
- `promotion_active`: run a discount promotion that increases demand

The simulation tracks:

- demand changes across the shift
- ingredient consumption and stockouts
- kitchen and server capacity constraints
- rating drift from service quality and pricing pressure
- labor, food, and reorder costs
- event-driven shocks such as large parties, inspections, delivery surges, and supplier delays

## Scenario Set

The environment includes 8 scenarios:

| ID | Difficulty | Core pressure |
|---|---|---|
| `weekday_lunch` | easy | predictable lunch peak and basic staffing discipline |
| `weekend_rush` | medium | damaged rating, surge demand, large party shock |
| `crisis_shift` | hard | doubled ingredient costs, low inventory, health inspection |
| `monsoon_delivery_crunch` | medium | rain-driven demand surge and supplier delay |
| `wedding_catering_chaos` | hard | premium evening demand, VIP visit, large party |
| `office_catering_lunch` | easy | concentrated corporate lunch throughput |
| `tourist_season_dinner` | medium | strong dinner demand with pricing sensitivity |
| `staff_shortage_recovery` | hard | under-staffed opening with weak inventory buffers |

## Reward and grading

The step reward is not just profit. It combines:

- normalized step profit
- customer rating trajectory
- service reliability

It then subtracts explicit penalties for:

- failed orders
- stockout failures
- capacity failures
- expensive emergency reorders
- inefficient labor usage when service remains weak

Final grading is deterministic and task-specific. The grader scores:

- `profit`
- `rating`
- `service`
- `satisfaction`
- `efficiency`

This makes the benchmark harder to game with one-dimensional strategies like permanent overpricing or overstaffing.

## Browser UI

The environment ships with a browser dashboard:

```bash
venv/bin/uvicorn app:app --host 0.0.0.0 --port 7860
```

Open:

```text
http://localhost:7860/play
```

The dashboard lets you:

- select any scenario
- inspect the current state
- toggle staff and menu availability
- change prices
- submit inventory reorders
- run steps manually
- inspect event logs, rewards, and final scores

## Local setup

```bash
git clone https://github.com/sheetalll28/Restaurant-manager-openenv
cd Restaurant-manager-openenv

python -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/pip install pytest
```

Run the API server:

```bash
venv/bin/uvicorn app:app --host 0.0.0.0 --port 7860
```

Run inference:

```bash
HF_TOKEN=your_token python inference.py
```

Run one task:

```bash
HF_TOKEN=your_token TASK=crisis_shift VERBOSE=true python inference.py
```

## Validation and tests

Validate the environment:

```bash
openenv validate
```

Run tests:

```bash
venv/bin/python -m pytest -q
```

## Deployment

To push this environment to your Hugging Face Space:

```bash
set -a
source .env
set +a
openenv push . --repo-id sheetallll21/restaurant-manager-openenv
```

If the token is valid but push fails with a permissions error, the token likely exists but does not have write access to your `sheetallll21` namespace.

## API surface

Main endpoints:

- `GET /` health/status
- `GET /play` browser dashboard
- `POST /reset` start a scenario
- `POST /step` execute one action
- `GET /state` fetch current observation
- `GET /tasks` list scenario metadata
- `GET /result` fetch final score/result snapshot

Example:

```python
import httpx

r = httpx.post("http://localhost:7860/reset", json={"task_id": "weekend_rush"})
state = r.json()["observation"]

action = {
    "staff_changes": {"Priya": True, "Sneha": True, "Arjun": True},
    "menu_changes": {"Butter Chicken": True},
    "price_adjustments": {"Paneer Tikka": 295},
    "reorder_inventory": {"paneer": 2.0},
    "promotion_active": False,
}

r = httpx.post("http://localhost:7860/step", json=action)
print(r.json()["reward"], r.json()["done"])
```

## Project structure

```text
.
├── app.py
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── README.md
├── ui/
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── env/
│   ├── environment.py
│   ├── graders.py
│   ├── models.py
│   └── tasks.py
├── server/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
└── tests/
    ├── test_environment.py
    └── test_models.py
```

## Submission checklist

Before submission, the high-signal checks are:

```bash
openenv validate
venv/bin/python -m pytest -q
openenv push . --repo-id sheetallll21/restaurant-manager-openenv
```

