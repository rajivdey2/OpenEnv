"""
inference.py
OpenEnv Submission Baseline Inference Script
"""
import os
import json
import requests
from openai import OpenAI
from typing import List, Optional

# Mandatory environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "dummy"

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
BENCHMARK = "DataCleanEnv"

SYSTEM_PROMPT = """You are a data cleaning agent interacting with a tabular dataset environment.

At each step you receive a JSON observation describing the current state of the data.
You must output ONLY a single valid JSON object matching this schema — no explanation, no markdown:

{
  "action_type": "<one of: fill_nulls | drop_nulls | deduplicate | normalize_column | cast_column | remove_orphans | flag_outliers | fix_outlier | inspect | join_inspect | merge_rows | done>",
  "column":      "<column name or null>",
  "table":       "<table name or null>",
  "value":       <fill value or null>,
  "strategy":    "<exact|fuzzy or null>",
  "format":      "<YYYY-MM-DD|PHONE|LOWERCASE|USD or null>",
  "target_type": "<int|float|str or null>",
  "row_id":      <integer or null>,
  "rule":        "<outlier rule string or null>",
  "ref_table":   "<reference table name or null>",
  "id1":         <integer or null>,
  "id2":         <integer or null>
}

Strategy:
1. Start with inspect to understand the data.
2. Fix issues systematically.
3. Call done when you believe all issues are resolved.
"""

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = str(error).replace("\n", " ").replace("\r", "") if error else "null"
    done_val = str(done).lower()
    # Ensure action string has no newlines to keep log on one line
    action_clean = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def run_task(task_id: int):
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    task_name = f"task_{task_id}"
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        r = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id})
        r.raise_for_status()
        data = r.json()
        episode_id = data["episode_id"]
        obs = data["observation"]
    except Exception as e:
        log_step(1, "reset_failed", 0.0, True, str(e))
        log_end(False, 1, 0.0, [0.0])
        return

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    steps_taken = 0
    score = 0.0
    rewards_list = []
    
    max_steps = 25
    done = False
    
    for step in range(1, max_steps + 1):
        if done:
            break
            
        messages.append({
            "role": "user",
            "content": f"Step {step} observation:\n{json.dumps(obs, indent=2, default=str)}"
        })
        
        action_error = None
        action_str = ""
        action_dict = None
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=300,
            )
            action_str = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": action_str})
            clean = action_str.replace("```json", "").replace("```", "").strip()
            action_dict = json.loads(clean)
        except Exception as e:
            action_error = str(e)
            action_dict = {"action_type": "done"}  # Recover by ending episode gracefully
            action_str = json.dumps(action_dict)
            
        reward = 0.0
        try:
            sr = requests.post(
                f"{ENV_URL}/step/{episode_id}",
                json=action_dict,
                timeout=10,
            )
            sr.raise_for_status()
            result = sr.json()
            obs = result["observation"]
            reward = result.get("reward", 0.0)
            done = result.get("done", True)
        except Exception as e:
            if not action_error:
                action_error = str(e)
            else:
                action_error += f" | {e}"
            done = True
            
        rewards_list.append(reward)
        steps_taken = step
        score = reward
        
        log_step(step=step, action=action_str, reward=reward, done=done, error=action_error)
        
    # Define success arbitrarily as having a score >= 0.5 at the end
    success = score >= 0.5
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards_list)

if __name__ == "__main__":
    for t_id in [1, 2, 3]:
        run_task(t_id)
