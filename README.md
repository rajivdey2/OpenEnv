 # DataCleanEnv — OpenEnv Data Cleaning Environment

## Overview

**DataCleanEnv** is an compliant environment where an AI agent
must identify and fix data quality issues in simulated tabular datasets.

Three tasks of increasing difficulty — from simple null-value filling to multi-table referential
integrity — each with continuous 0.0–1.0 rewards and rich partial-progress signals at every step.

---

## Motivation

Data cleaning is one of the most time-consuming tasks in any data engineering or ML pipeline.
Every data engineer encounters nulls, duplicates, type mismatches, and broken foreign keys daily.
This environment tests whether an LLM agent can reason about tabular data structure,
plan a sequence of cleaning operations, and recover from mistakes — a real-world agentic skill.

---

## Environment Description

The agent interacts with the environment through three endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode; returns initial observation + `episode_id` |
| `/step/{episode_id}` | POST | Send a `CleaningAction`; returns next observation + reward |
| `/state/{episode_id}` | GET | Read episode metadata (step count, actions taken) |
| `/health` | GET | Liveness probe |

---

## Action Space

All actions are structured JSON matching the `CleaningAction` Pydantic model.

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `inspect` | `column?`, `table?` | View column stats or full table summary |
| `fill_nulls` | `column`, `value?` | Fill null values in a column |
| `drop_nulls` | `column?` | Drop rows with nulls |
| `deduplicate` | `strategy` (`exact`\|`fuzzy`) | Remove duplicate rows |
| `merge_rows` | `id1`, `id2` | Merge two near-duplicate rows |
| `normalize_column` | `column`, `format` | Normalize dates, phones, or currency |
| `cast_column` | `column`, `target_type`, `table?` | Fix column data types |
| `remove_orphans` | `column`, `ref_table?` | Remove rows with broken foreign keys |
| `flag_outliers` | `column`, `rule?`, `table?` | Report outlier rows |
| `fix_outlier` | `column`, `row_id`, `value`, `table?` | Correct a specific outlier |
| `join_inspect` | — | Check referential integrity across tables |
| `done` | — | Signal episode completion |

---

## Observation Space

Each step returns a `CleaningObservation`:

```json
{
  "success": true,
  "message": "Filled 5 nulls in 'age' with value '0'.",
  "table_preview": [...],
  "issues_remaining": 12,
  "partial_score": 0.42,
  "done": false
}
```

---

## Tasks

### Task 1: Fix Null Values (Easy)

A 30-row table with deliberate nulls spread across `age`, `salary`, `department`, `city`, `score`.
The agent must discover which columns have nulls (via `inspect`) and fill them with appropriate defaults.

**Actions available:** `inspect`, `fill_nulls`, `drop_nulls`, `done`

**Score:** `1 - (nulls_remaining / nulls_initial)` — averaged across affected columns.

**Expected baseline score:** ~0.75–0.90

---

### Task 2: Normalize & Deduplicate (Medium)

A 23-row table with:
- Exact duplicate rows
- Near-duplicate rows (same person, different email capitalisation)
- Mixed date formats (`Jan 5 2024`, `15/03/2023`, `2022-07-20`, …)
- Mixed phone formats (`9876543210`, `+91-98765-43210`, `(987) 654-3210`, …)

**Score:** `0.4 × dedup + 0.3 × date_normalisation + 0.3 × phone_normalisation`

**Expected baseline score:** ~0.50–0.65

---

### Task 3: Schema Alignment + Referential Integrity (Hard)

Two related tables (`customers` + `orders`) with:
- `age` stored as `float` → should be `int`; outlier ages (`999`, `-5`, `150`)
- `price` stored as currency strings (`"$12.50"`, `"Rs.300"`) → should be `float`
- Orphaned orders (4 orders referencing non-existent customer IDs)
- Mixed currencies (`INR`, `EUR`, `GBP`) → should all be `USD`
- Balance outliers (`99999`, `9999`)

**Score:** `0.25 × type_correctness + 0.25 × referential_integrity + 0.25 × outlier_handling + 0.25 × currency_consistency`

**Expected baseline score:** ~0.30–0.50

---

## Reward Function

```
reward = base_score - loop_penalty - destructive_penalty

loop_penalty      = 0.05 × (consecutive_same_action_count - 2)  if count ≥ 3
destructive_penalty = 0.05  if drop_nulls is called
```

All rewards are clipped to [0.0, 1.0]. Partial credit is given at every step.

---

## Setup & Usage

### Local (Docker)

```bash
git clone https://huggingface.co/spaces/<your-username>/data-clean-env
cd data-clean-env
docker build -t data-clean-env .
docker run -p 8000:8000 data-clean-env
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Hugging Face Spaces

The Space runs automatically on push. Set `OPENAI_API_KEY` as a Space secret to enable the baseline demo.

---

## Running the Baseline

```bash
export OPENAI_API_KEY=sk-...
export ENV_URL=http://localhost:8000      # or your HF Space URL
pip install openai requests
python baseline/run_baseline.py
```

---

## Baseline Scores

| Task | Difficulty | Mean Score (gpt-4o-mini) |
|------|------------|--------------------------|
| 1    | Easy       | ~0.82                    |
| 2    | Medium     | ~0.57                    |
| 3    | Hard       | ~0.38                    |

*(Scores updated after official baseline run)*

---

## Contributing

Issues and PRs welcome. Please ensure any new tasks include:
- A `generate_dirty_data()` function
- A `get_issues()` snapshot function
- An `execute_action()` dispatcher
- A `grade()` function returning a float in [0, 1]
- A `count_remaining_issues()` helper
