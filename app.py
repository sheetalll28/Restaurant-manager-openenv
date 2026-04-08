"""
FastAPI server for the Restaurant Manager OpenEnv.

Exposes the environment over HTTP:
  GET  /           → health check (returns 200 + status)
  GET  /health     → alias for health check
  POST /reset      → reset environment with task_id
  POST /step       → submit action, get observation
  GET  /state      → current state
  GET  /tasks      → list available tasks
  GET  /result     → final shift result (after done=True)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env.environment import RestaurantEnv
from env.graders import grade
from env.models import AgentAction, RestaurantState


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
    task_id: str



env = RestaurantEnv()
current_task_id: str | None = None
VALID_TASKS = ["weekday_lunch", "weekend_rush", "crisis_shift"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Restaurant Manager OpenEnv",
    description="AI restaurant shift management environment — OpenEnv compatible",
    version="1.1.0",
    lifespan=lifespan,
)

# Allow cross-origin requests (needed for HF Space iframe interactions)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health_check():
    """Health check — returns 200 if server is running. Required by OpenEnv validator."""
    return {
        "status": "ok",
        "environment": "restaurant-manager",
        "version": "1.1.0",
        "tasks": VALID_TASKS,
    }


@app.get("/health")
async def health_check_alias():
    """Alias for health check — some validators ping /health."""
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest = Body(default=ResetRequest())):
    """
    Reset the environment with the specified task.
    Returns the initial observation.
    """
    global current_task_id

    if request.task_id not in VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{request.task_id}'. Available: {VALID_TASKS}",
        )

    try:
        observation = env.reset(request.task_id)
        current_task_id = request.task_id
        return ResetResponse(observation=observation, task_id=request.task_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(action: AgentAction):
    """
    Execute one step in the environment.
    Call POST /reset first to initialize.
    """
    if current_task_id is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first.",
        )

    try:
        observation, reward, done, info = env.step(action)

        if done:
            result = env.get_result()
            grade_report = grade(current_task_id, result)
            info["final_score"] = round(grade_report["final_score"] / 100.0, 4)
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
    """Return the current environment state."""
    if current_task_id is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first.",
        )
    try:
        return {"observation": env.state().model_dump(), "task_id": current_task_id}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/result")
async def get_result():
    """Get the final shift result. Call after done=True."""
    if current_task_id is None:
        raise HTTPException(status_code=400, detail="No episode in progress.")
    try:
        result = env.get_result()
        grade_report = grade(current_task_id, result)
        return {
            "task_id": current_task_id,
            "shift_result": result.model_dump(),
            "final_score": round(grade_report["final_score"] / 100.0, 4),
            "pillar_scores": grade_report["pillar_scores"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
async def list_tasks():
    """List all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "id": "weekday_lunch",
                "name": "Weekday Lunch Service",
                "difficulty": "easy",
                "description": "Stable weekday lunch with normal demand and full inventory.",
                "targets": {"profit": 8000, "rating": 4.2, "service_rate": 0.80},
            },
            {
                "id": "weekend_rush",
                "name": "Weekend Festival Rush",
                "difficulty": "medium",
                "description": "High demand surge, low starting rating, large party at step 4, best chef unavailable.",
                "targets": {"profit": 12000, "rating": 4.0, "service_rate": 0.75},
            },
            {
                "id": "crisis_shift",
                "name": "Crisis Management Shift",
                "difficulty": "hard",
                "description": "Doubled ingredient costs, 40% inventory, health inspection at step 8, competitor pressure.",
                "targets": {"profit": 5000, "rating": 4.0, "service_rate": 0.70},
            },
        ]
    }


if __name__ == "__main__":

    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)