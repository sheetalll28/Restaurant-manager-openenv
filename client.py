from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import AgentAction, RestaurantState


class RestaurantManagerEnv(EnvClient[AgentAction, RestaurantState, State]):
    def _step_payload(self, action: AgentAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[RestaurantState]:
        observation = RestaurantState.model_validate(payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        observation = payload.get("observation") or {}
        return State(
            episode_id=payload.get("task_id"),
            step_count=observation.get("step", 0),
        )

