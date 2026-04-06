"""
baseline/run_baseline.py

Runs a baseline LLM agent against all 3 DataCleanEnv tasks.
Uses OpenAI API client (compatible with any OpenAI-compatible endpoint).

Usage:
    export OPENAI_API_KEY=sk-...
    export ENV_URL=http://localhost:8000     # default
    python baseline/run_baseline.py
"""
import os
import sys
import json
import time
import requests
from openai import OpenAI
from dotenv import load_dotenv
 
load_dotenv()  # ← add this

BASE_URL = os.environ.get("ENV_URL", "http://localhost:8000")
client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
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
2. Fix issues systematically — nulls first, then duplicates/types/outliers.
3. Call done when you believe all issues are resolved.
4. Avoid repeating the same action more than twice in a row (incurs penalty).
"""


def run_episode(task_id: int, max_steps: int = 25) -> float:
    """Run a single episode for the given task. Returns the final reward."""

    # ── Reset ────────────────────────────────────────────────────────────────
    r = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id})
    r.raise_for_status()
    data       = r.json()
    episode_id = data["episode_id"]
    obs        = data["observation"]

    messages      = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_reward  = 0.0

    for step_num in range(1, max_steps + 1):
        # Build user message
        messages.append({
            "role":    "user",
            "content": f"Step {step_num} observation:\n{json.dumps(obs, indent=2, default=str)}",
        })

        # ── LLM call ─────────────────────────────────────────────────────────
        try:
            response    = client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=messages,
                temperature=0.1,
                max_tokens=300,
            )
            action_str  = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": action_str})
        except Exception as e:
            print(f"  [LLM error] {e}")
            break

        # ── Parse action ──────────────────────────────────────────────────────
        try:
            # Strip optional markdown fences the model might add
            clean = action_str.replace("```json", "").replace("```", "").strip()
            action = json.loads(clean)
        except json.JSONDecodeError as e:
            print(f"  [JSON parse error] {e} — raw: {action_str[:120]}")
            # Inject a recover-inspect so the episode continues
            action = {"action_type": "inspect"}

        # ── Step environment ──────────────────────────────────────────────────
        try:
            sr = requests.post(
                f"{BASE_URL}/step/{episode_id}",
                json=action,
                timeout=10,
            )
            sr.raise_for_status()
            result       = sr.json()
            obs          = result["observation"]
            total_reward = result["reward"]

            print(
                f"  step={step_num:02d}  action={action.get('action_type','?'):<20}"
                f"  score={total_reward:.3f}  issues_left={obs['issues_remaining']}"
            )

            if result["done"]:
                break
        except requests.RequestException as e:
            print(f"  [HTTP error] {e}")
            break

        time.sleep(0.3)   # be nice to the server

    return total_reward


def main():
    print(f"\nDataCleanEnv Baseline — target: {BASE_URL}\n{'='*50}")
    results = {}

    for task_id in [1, 2, 3]:
        print(f"\n── Task {task_id} ──────────────────────────────────────")
        scores = []
        for run in range(1, 4):       # 3 runs per task
            print(f"  Run {run}/3:")
            score = run_episode(task_id)
            scores.append(score)
            print(f"  → Final score: {score:.3f}\n")
            time.sleep(1)

        results[f"task_{task_id}"] = {
            "scores": scores,
            "mean":   round(sum(scores) / len(scores), 3),
        }

    print("\n" + "="*50)
    print("BASELINE RESULTS")
    print("="*50)
    for key, val in results.items():
        print(f"{key}: scores={val['scores']}  mean={val['mean']:.3f}")

    # Write results to JSON for reproducibility
    out_path = os.path.join(os.path.dirname(__file__), "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()