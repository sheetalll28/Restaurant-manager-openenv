"""
FastAPI server for the Restaurant Manager OpenEnv.

Exposes the environment over HTTP for:
  - HF Space deployment
  - openenv validate
  - Remote inference

Endpoints:
  GET  /           → health check
  POST /reset      → reset environment with a task_id
  POST /step       → submit an action, get next state
  GET  /state      → get current state
  GET  /tasks      → list available tasks
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from env.environment import RestaurantEnv
from env.graders import grade
from env.models import AgentAction, RestaurantState

# ── Request / Response schemas ────────────────────────────────────────────


class ResetRequest(BaseModel):
    task_id: str = Field(
        default="weekday_lunch",
        description="Task to run: weekday_lunch, weekend_rush, or crisis_shift",
    )


class StepResponse(BaseModel):
    observation: RestaurantState
    reward: float
    done: bool
    info: dict


class ResetResponse(BaseModel):
    observation: RestaurantState


class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: str
    description: str


# ── Global environment instance ───────────────────────────────────────────

env = RestaurantEnv()
current_task_id: str | None = None


# ── App setup ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    yield


app = FastAPI(
    title="Restaurant Manager OpenEnv",
    description="AI restaurant shift management environment — OpenEnv compatible",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.get("/")
async def health_check():
    """Health check endpoint. Returns 200 if the server is running."""
    return {"status": "ok", "environment": "restaurant-manager", "version": "1.0.0"}


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest):
    """
    Reset the environment with the specified task.

    Returns the initial observation.
    """
    global current_task_id

    valid_tasks = ["weekday_lunch", "weekend_rush", "crisis_shift"]
    if request.task_id not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{request.task_id}'. Available: {valid_tasks}",
        )

    try:
        observation = env.reset(request.task_id)
        current_task_id = request.task_id
        return ResetResponse(observation=observation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(action: AgentAction):
    """
    Execute one step in the environment.

    Accepts an AgentAction and returns the new observation, reward, done flag,
    and info dict.
    """
    if current_task_id is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first.",
        )

    try:
        observation, reward, done, info = env.step(action)

        # If episode is done, include final grade in info
        if done:
            result = env.get_result()
            grade_report = grade(current_task_id, result)
            info["final_score"] = grade_report["final_score"] / 100.0  # normalize to [0,1]
            info["pillar_scores"] = grade_report["pillar_scores"]
            info["shift_result"] = result.model_dump()

        return StepResponse(
            observation=observation,
            reward=round(reward, 4),
            done=done,
            info=info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state():
    """
    Return the current environment state.

    Must call POST /reset before calling this endpoint.
    """
    if current_task_id is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first.",
        )

    try:
        current_state = env.state()
        return {"observation": current_state.model_dump()}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
async def list_tasks():
    """List all available tasks with their descriptions."""
    return {
        "tasks": [
            {
                "id": "weekday_lunch",
                "name": "Weekday Lunch Service",
                "difficulty": "easy",
                "description": "Stable weekday lunch with normal demand and full inventory.",
            },
            {
                "id": "weekend_rush",
                "name": "Weekend Festival Rush",
                "difficulty": "medium",
                "description": "High demand surge with low starting rating and large party event.",
            },
            {
                "id": "crisis_shift",
                "name": "Crisis Management Shift",
                "difficulty": "hard",
                "description": "Doubled ingredient costs, low inventory, health inspection, competitor pressure.",
            },
        ]
    }


# ── Run server ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="localhost", port=port)
