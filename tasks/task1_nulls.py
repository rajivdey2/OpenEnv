"""
Task 1: Fix Null Values (Easy)

The agent receives a table with nullable columns and must fill or drop
nulls to achieve a clean dataset. Grader scores based on fraction of
nulls resolved.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any


# --------------------------------------------------------------------------- #
#  Data generation                                                             #
# --------------------------------------------------------------------------- #

COLUMN_DEFAULTS: Dict[str, Any] = {
    "age":        0,
    "salary":     0.0,
    "department": "Unknown",
    "city":       "Unknown",
    "score":      0.0,
}

def generate_dirty_data() -> Dict[str, pd.DataFrame]:
    """Return a dict with one DataFrame called 'main' that has deliberate nulls."""
    np.random.seed(42)
    n = 30

    df = pd.DataFrame({
        "id":         range(1, n + 1),
        "name":       [f"Person_{i}" for i in range(1, n + 1)],
        "age":        [np.nan if i % 5 == 0 else float(20 + i) for i in range(n)],
        "salary":     [np.nan if i % 4 == 0 else float(30000 + i * 500) for i in range(n)],
        "department": [None if i % 6 == 0 else f"Dept_{i % 4}" for i in range(n)],
        "city":       [None if i % 7 == 0 else f"City_{i % 5}" for i in range(n)],
        "score":      [np.nan if i % 8 == 0 else round(float(i) / n, 2) for i in range(n)],
    })

    return {"main": df}


def get_issues(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Snapshot of ground-truth issues at the time of reset.
    Records per-column null counts and their correct fill values.
    """
    df = data["main"]
    issues = {}
    for col, default in COLUMN_DEFAULTS.items():
        null_idx = df.index[df[col].isna()].tolist()
        if null_idx:
            issues[col] = {
                "null_indices": null_idx,
                "initial_count": len(null_idx),
                "correct_value": default,
            }
    return issues


# --------------------------------------------------------------------------- #
#  Action execution                                                            #
# --------------------------------------------------------------------------- #

def execute_action(action, data: Dict[str, pd.DataFrame]) -> Tuple[str, bool]:
    """
    Apply a CleaningAction to the in-memory data.
    Returns (message, success).
    """
    df = data.get("main")
    if df is None:
        return "Table 'main' not found.", False

    atype = action.action_type.value

    # ── inspect ──────────────────────────────────────────────────────────────
    if atype == "inspect":
        col = action.column
        if col and col in df.columns:
            null_count = int(df[col].isna().sum())
            sample = df[col].dropna().head(5).tolist()
            return (
                f"Column '{col}': {null_count} nulls out of {len(df)} rows. "
                f"Sample values: {sample}",
                True,
            )
        # inspect whole table
        summary = {c: int(df[c].isna().sum()) for c in df.columns}
        return f"Null counts per column: {summary}", True

    # ── fill_nulls ───────────────────────────────────────────────────────────
    if atype == "fill_nulls":
        col = action.column
        val = action.value
        if col is None or col not in df.columns:
            return f"Column '{col}' not found.", False
        if val is None:
            val = COLUMN_DEFAULTS.get(col, 0)
        before = int(df[col].isna().sum())
        df[col] = df[col].fillna(val)
        data["main"] = df
        after = int(df[col].isna().sum())
        return f"Filled {before - after} nulls in '{col}' with value '{val}'.", True

    # ── drop_nulls ───────────────────────────────────────────────────────────
    if atype == "drop_nulls":
        col = action.column
        if col and col not in df.columns:
            return f"Column '{col}' not found.", False
        before = len(df)
        if col:
            df = df.dropna(subset=[col])
        else:
            df = df.dropna()
        data["main"] = df.reset_index(drop=True)
        dropped = before - len(data["main"])
        target = f"'{col}'" if col else "all columns"
        return f"Dropped {dropped} rows with nulls in {target}.", True

    # ── done ─────────────────────────────────────────────────────────────────
    if atype == "done":
        return "Agent signalled completion.", True

    return f"Action '{atype}' is not supported in Task 1.", False


# --------------------------------------------------------------------------- #
#  Grader                                                                      #
# --------------------------------------------------------------------------- #

def grade(data: Dict[str, pd.DataFrame], issues: Dict[str, Any]) -> float:
    """
    Score = 1 - (nulls_remaining / nulls_initial)
    Averaged across all columns that had issues.
    """
    if not issues:
        return 1.0

    df = data["main"]
    scores = []
    for col, info in issues.items():
        initial = info["initial_count"]
        if initial == 0:
            scores.append(1.0)
            continue
        # Count how many of the originally-null rows are still null
        idx = [i for i in info["null_indices"] if i < len(df)]
        remaining = int(df.loc[idx, col].isna().sum()) if idx else 0
        col_score = 1.0 - (remaining / initial)
        scores.append(col_score)

    return round(sum(scores) / len(scores), 4)


def count_remaining_issues(data: Dict[str, pd.DataFrame]) -> int:
    """Total null cells still present in the main table."""
    df = data["main"]
    return int(df[list(COLUMN_DEFAULTS.keys())].isna().sum().sum())