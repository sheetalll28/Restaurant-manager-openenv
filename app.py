"""
FastAPI server for the Restaurant Manager OpenEnv.

API:
  GET  /           -> health check
  GET  /play       -> browser UI
  GET  /health     -> alias for health check
  POST /reset      -> reset environment with task_id
  POST /step       -> submit action and receive observation
  GET  /state      -> current state
  GET  /tasks      -> task metadata
  GET  /result     -> final score/result snapshot
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from env.environment import RestaurantEnv
from env.graders import grade
from env.models import AgentAction, RestaurantState
from env.tasks import TASK_SPECS, list_task_metadata


class ResetRequest(BaseModel):
    task_id: str = Field(default="weekday_lunch", description="Task to run")


class StepResponse(BaseModel):
    observation: RestaurantState
    reward: float
    done: bool
    info: dict


class ResetResponse(BaseModel):
    observation: RestaurantState
    task_id: str


BASE_DIR = Path(__file__).parent
UI_DIR = BASE_DIR / "ui"

env = RestaurantEnv()
current_task_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Restaurant Manager OpenEnv",
    description="AI restaurant shift management environment with browser UI",
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if UI_DIR.exists():
    app.mount("/assets", StaticFiles(directory=UI_DIR), name="ui-assets")
    app.mount("/web/assets", StaticFiles(directory=UI_DIR), name="ui-assets-web")


def _ui_response() -> FileResponse:
    index_path = UI_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found.")
    return FileResponse(index_path)


@app.get("/")
async def play_ui_root():
    return _ui_response()


@app.get("/web")
@app.get("/web/")
async def play_ui_web_root():
    return _ui_response()


@app.get("/play")
async def play_ui():
    return _ui_response()


@app.get("/web/play")
async def play_ui_web():
    return _ui_response()


@app.get("/status")
async def status():
    return {
        "status": "ok",
        "environment": "restaurant-manager",
        "version": "1.2.0",
        "tasks": list(TASK_SPECS.keys()),
        "ui": "/play",
    }


@app.get("/web/status")
async def status_web():
    return await status()


@app.get("/health")
async def health_check_alias():
    return {"status": "ok"}


@app.get("/web/health")
async def health_check_web():
    return await health_check_alias()


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest | None = Body(default=None)):
    global current_task_id

    requested_task_id = request.task_id if request is not None else "weekday_lunch"
    if requested_task_id not in TASK_SPECS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{requested_task_id}'. Available: {list(TASK_SPECS.keys())}",
        )

    try:
        observation = env.reset(requested_task_id)
        current_task_id = requested_task_id
        return ResetResponse(observation=observation, task_id=requested_task_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step", response_model=StepResponse)
async def step(action: AgentAction):
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
        return StepResponse(observation=observation, reward=reward, done=done, info=info)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state")
async def get_state():
    if current_task_id is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first.",
        )
    try:
        return {"observation": env.state().model_dump(), "task_id": current_task_id}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/result")
async def get_result():
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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/tasks")
async def list_tasks():
    return {"tasks": list_task_metadata()}


def main():
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
