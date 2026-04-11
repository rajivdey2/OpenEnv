"""
inference.py — OpenEnv DataCleanEnv Inference Script

Starts the FastAPI server in a background thread, then runs the LLM agent
against all 3 tasks. All LLM calls go through API_BASE_URL + API_KEY
as injected by the validator.

Required env vars (injected by validator):
    API_KEY        API key for LLM proxy
    API_BASE_URL   LiteLLM proxy endpoint
    MODEL_NAME     Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
"""

import os
import sys
import json
import time
import threading
import textwrap
import requests
from typing import List, Optional

# ── Add project root to path ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

# ── Env vars — strictly from injected environment ────────────────────────────
API_KEY      = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_PORT     = int(os.getenv("ENV_PORT", "8000"))
ENV_URL      = f"http://127.0.0.1:{ENV_PORT}"
BENCHMARK    = "data-clean-env"
MAX_STEPS    = 20
SUCCESS_THRESHOLD = 0.5

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── Start FastAPI server in background thread ─────────────────────────────────

def _start_server():
    import uvicorn
    from server.app import app
    uvicorn.run(app, host="127.0.0.1", port=ENV_PORT, log_level="error")

def start_server_and_wait(timeout: int = 30):
    t = threading.Thread(target=_start_server, daemon=True)
    t.start()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=2)
            if r.status_code == 200:
                print(f"[DEBUG] Server ready at {ENV_URL}", flush=True)
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Server did not start within {timeout}s")

# ── Logging (exact required format) ──────────────────────────────────────────

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    action_safe = str(action).replace("\n", " ")[:120]
    print(f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a data cleaning agent interacting with a tabular dataset environment.
    At each step you receive a JSON observation. Output ONLY a single valid JSON object — no explanation, no markdown.

    Schema:
    {
      "action_type": "<fill_nulls|drop_nulls|deduplicate|normalize_column|cast_column|remove_orphans|flag_outliers|fix_outlier|inspect|join_inspect|merge_rows|done>",
      "column": "<column name or null>",
      "table": "<table name or null>",
      "value": <fill value or null>,
      "strategy": "<exact|fuzzy or null>",
      "format": "<YYYY-MM-DD|PHONE|USD or null>",
      "target_type": "<int|float|str or null>",
      "row_id": <integer or null>,
      "rule": "<rule string or null>",
      "ref_table": "<table name or null>",
      "id1": <integer or null>,
      "id2": <integer or null>
    }

    Strategy:
    1. Start with inspect to understand the data.
    2. Fix issues: nulls first, then duplicates, then type errors, then outliers.
    3. Call done when issues_remaining = 0 or no more progress is possible.
    4. Never repeat the same action more than twice in a row.
""").strip()

# ── LLM call ─────────────────────────────────────────────────────────────────

def llm_call(messages: list, max_retries: int = 5) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=300,
                stream=False,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower() or "RESOURCE_EXHAUSTED" in err:
                wait = 15 * (attempt + 1)
                for part in err.split():
                    try:
                        c = float(part.rstrip("s.,"))
                        if 1 < c < 300:
                            wait = c + 5
                            break
                    except ValueError:
                        pass
                print(f"[DEBUG] Rate limit, waiting {wait:.0f}s (attempt {attempt+1})", flush=True)
                time.sleep(wait)
            else:
                print(f"[DEBUG] LLM error: {err[:150]}", flush=True)
                return None
    return None

# ── Single episode ────────────────────────────────────────────────────────────

def run_episode(task_id: int, task_name: str) -> float:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        r = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id}, timeout=15)
        r.raise_for_status()
        data       = r.json()
        episode_id = data["episode_id"]
        obs        = data["observation"]
        messages   = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS + 1):
            messages.append({
                "role":    "user",
                "content": f"Step {step}:\n{json.dumps(obs, indent=2, default=str)}",
            })

            action_str = llm_call(messages)
            error_msg  = None

            if action_str is None:
                action_str = '{"action_type": "inspect"}'
                error_msg  = "llm_fallback_inspect"

            messages.append({"role": "assistant", "content": action_str})

            try:
                clean  = action_str.replace("```json", "").replace("```", "").strip()
                action = json.loads(clean)
            except json.JSONDecodeError as e:
                action    = {"action_type": "inspect"}
                error_msg = f"json_parse_error:{str(e)[:40]}"

            reward = 0.0
            done   = False
            try:
                sr = requests.post(f"{ENV_URL}/step/{episode_id}", json=action, timeout=15)
                if sr.status_code == 404:
                    done = True
                    log_step(step, action.get("action_type", "?"), reward, done, "episode_not_found")
                    rewards.append(reward)
                    steps_taken = step
                    break
                sr.raise_for_status()
                result = sr.json()
                obs    = result["observation"]
                reward = float(result.get("reward", 0.0))
                done   = bool(result.get("done", False))
            except requests.RequestException as e:
                error_msg = f"http_error:{str(e)[:40]}"
                done = True

            log_step(step, action.get("action_type", "?"), reward, done, error_msg)
            rewards.append(reward)
            steps_taken = step

            if done:
                break

            time.sleep(0.3)

        score   = rewards[-1] if rewards else 0.0
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        score   = rewards[-1] if rewards else 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    start_server_and_wait()

    tasks = [
        (1, "fix-null-values"),
        (2, "normalize-and-deduplicate"),
        (3, "schema-alignment"),
    ]

    all_scores = {}
    for task_id, task_name in tasks:
        print(f"\n[DEBUG] === Task {task_id}: {task_name} ===", flush=True)
        score = run_episode(task_id, task_name)
        all_scores[task_name] = round(score, 3)
        time.sleep(2)

    print("\n[DEBUG] === FINAL RESULTS ===", flush=True)
    for name, sc in all_scores.items():
        print(f"[DEBUG] {name}: {sc:.3f}", flush=True)


if __name__ == "__main__":
    main()
