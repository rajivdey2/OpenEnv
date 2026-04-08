"""
inference.py — OpenEnv DataCleanEnv Inference Script
=====================================================
MANDATORY env vars:
    API_BASE_URL   LLM API endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier   (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       HuggingFace / API key

STDOUT FORMAT (strictly enforced):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import time
import textwrap
import requests
from typing import List, Optional

from openai import OpenAI

# ── Environment variables ────────────────────────────────────────────────────
API_KEY      = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:8000")
BENCHMARK    = "data-clean-env"

MAX_STEPS    = 20
SUCCESS_THRESHOLD = 0.5

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── Logging helpers (exact format required) ──────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Sanitise action string — no newlines allowed on a single log line
    action_safe = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a data cleaning agent interacting with a tabular dataset environment.
    At each step you receive a JSON observation describing the current state of the data.
    Output ONLY a single valid JSON object — no explanation, no markdown fences.

    Schema:
    {
      "action_type": "<fill_nulls|drop_nulls|deduplicate|normalize_column|cast_column|remove_orphans|flag_outliers|fix_outlier|inspect|join_inspect|merge_rows|done>",
      "column":      "<column name or null>",
      "table":       "<table name or null>",
      "value":       <fill value or null>,
      "strategy":    "<exact|fuzzy or null>",
      "format":      "<YYYY-MM-DD|PHONE|USD or null>",
      "target_type": "<int|float|str or null>",
      "row_id":      <integer or null>,
      "rule":        "<rule string or null>",
      "ref_table":   "<table name or null>",
      "id1":         <integer or null>,
      "id2":         <integer or null>
    }

    Strategy:
    1. Start with inspect to understand the data.
    2. Fix issues systematically: nulls → duplicates → type errors → outliers → done.
    3. Call done when all issues_remaining = 0 or no more progress is possible.
    4. Never repeat the same action more than twice in a row.
""").strip()

# ── LLM call with retry ──────────────────────────────────────────────────────

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
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "rate" in err.lower():
                wait = 15 * (attempt + 1)
                for part in err.split():
                    try:
                        c = float(part.rstrip("s.,"))
                        if 1 < c < 300:
                            wait = c + 5
                            break
                    except ValueError:
                        pass
                print(f"[DEBUG] Rate limit — waiting {wait:.0f}s (attempt {attempt+1})", flush=True)
                time.sleep(wait)
            else:
                print(f"[DEBUG] LLM error: {err[:120]}", flush=True)
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
        # reset
        r = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id}, timeout=15)
        r.raise_for_status()
        data       = r.json()
        episode_id = data["episode_id"]
        obs        = data["observation"]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS + 1):
            messages.append({
                "role":    "user",
                "content": f"Step {step} observation:\n{json.dumps(obs, indent=2, default=str)}",
            })

            action_str = llm_call(messages)
            error_msg  = None

            if action_str is None:
                action_str = '{"action_type": "inspect"}'
                error_msg  = "llm_call returned None, using fallback inspect"

            messages.append({"role": "assistant", "content": action_str})

            # parse action
            try:
                clean  = action_str.replace("```json", "").replace("```", "").strip()
                action = json.loads(clean)
            except json.JSONDecodeError as e:
                action    = {"action_type": "inspect"}
                error_msg = f"json_parse_error: {str(e)[:60]}"

            # step env
            reward = 0.0
            done   = False
            try:
                sr = requests.post(
                    f"{ENV_URL}/step/{episode_id}",
                    json=action,
                    timeout=15,
                )
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
                error_msg = f"http_error: {str(e)[:60]}"
                done = True

            log_step(
                step=step,
                action=action.get("action_type", str(action)),
                reward=reward,
                done=done,
                error=error_msg,
            )
            rewards.append(reward)
            steps_taken = step

            if done:
                break

            time.sleep(0.4)

        score   = rewards[-1] if rewards else 0.0   # final score = last reward
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        score   = rewards[-1] if rewards else 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ── Main: run all 3 tasks ─────────────────────────────────────────────────────

def main():
    tasks = [
        (1, "fix-null-values"),
        (2, "normalize-and-deduplicate"),
        (3, "schema-alignment"),
    ]

    all_scores = {}
    for task_id, task_name in tasks:
        print(f"\n[DEBUG] === Running task {task_id}: {task_name} ===", flush=True)
        score = run_episode(task_id, task_name)
        all_scores[task_name] = score
        time.sleep(2)

    print("\n[DEBUG] === FINAL RESULTS ===", flush=True)
    for name, sc in all_scores.items():
        print(f"[DEBUG] {name}: {sc:.3f}", flush=True)


if __name__ == "__main__":
    main()