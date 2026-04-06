"""
server/app.py — FastAPI server exposing the OpenEnv endpoints.

Endpoints:
  POST /reset              → start a new episode
  POST /step/{episode_id}  → take an action
  GET  /state/{episode_id} → read current episode state
  GET  /health             → liveness probe (for UptimeRobot etc.)
"""
import sys
import os

# Make sure the project root is on the Python path when running from /server
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import CleaningAction, CleaningObservation, EpisodeState
from environment import DataCleanEnvironment

app = FastAPI(
    title="DataCleanEnv",
    description=(
        "OpenEnv-compliant data cleaning environment. "
        "An AI agent must identify and fix data quality issues "
        "(nulls, duplicates, type errors, schema mismatches) in simulated tabular datasets."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory episode store  {episode_id: DataCleanEnvironment}
envs: dict[str, DataCleanEnvironment] = {}


# --------------------------------------------------------------------------- #
#  Routes                                                                      #
# --------------------------------------------------------------------------- #

@app.post("/reset")
def reset(task_id: int = 1):
    """
    Start a new episode for the given task.
    Returns the initial observation and a unique episode_id.
    """
    if task_id not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="task_id must be 1, 2, or 3.")

    env = DataCleanEnvironment(task_id=task_id)
    obs = env.reset()
    episode_id = env.state.episode_id
    envs[episode_id] = env

    return {
        "observation": obs.model_dump(),
        "episode_id":  episode_id,
    }


@app.post("/step/{episode_id}")
def step(episode_id: str, action: CleaningAction):
    """
    Apply an action to the environment and return the next observation.
    The episode is deleted from memory once done=True.
    """
    env = envs.get(episode_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Episode '{episode_id}' not found. Call /reset first.",
        )

    obs = env.step(action)

    response = {
        "observation": obs.model_dump(),
        "reward":      obs.partial_score,
        "done":        obs.done,
        "info":        {
            "step":     env.state.step_count,
            "max_steps": env.state.max_steps,
        },
    }

    if obs.done:
        del envs[episode_id]

    return response


@app.get("/state/{episode_id}")
def state(episode_id: str):
    """Return metadata about a running episode."""
    env = envs.get(episode_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Episode '{episode_id}' not found.",
        )
    return env.state.model_dump()


@app.get("/health")
def health():
    """Liveness probe — used by UptimeRobot to keep the Space awake."""
    return {"status": "ok", "active_episodes": len(envs)}


@app.get("/")
def root():
    return {
        "name":    "DataCleanEnv",
        "version": "0.1.0",
        "docs":    "/docs",
        "tasks":   [1, 2, 3],
    }