"""
Task 2: Normalize & Deduplicate (Medium)

The agent receives a table with:
  - Exact duplicate rows
  - Near-duplicate rows (same person, different email capitalisation)
  - Inconsistent date formats  (e.g. "Jan 5 2024" vs "2024-01-05")
  - Inconsistent phone formats (e.g. "9876543210" vs "+91-98765-43210")

Score = 0.4 × dedup_score + 0.3 × date_score + 0.3 × phone_score
"""
import re
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from datetime import datetime


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

TARGET_DATE_FMT  = "%Y-%m-%d"
TARGET_PHONE_FMT = "XXXXXXXXXX"          # 10-digit string, no spaces/dashes


def _normalize_phone(val: str) -> str:
    """Strip everything except digits and return last 10."""
    digits = re.sub(r"\D", "", str(val))
    return digits[-10:] if len(digits) >= 10 else digits


def _is_normalized_phone(val) -> bool:
    if pd.isna(val):
        return True
    return bool(re.fullmatch(r"\d{10}", str(val)))


def _parse_date(val) -> str | None:
    """Try to parse a date string into YYYY-MM-DD. Return None on failure."""
    if pd.isna(val):
        return None
    val = str(val).strip()
    formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y",
        "%b %d %Y", "%B %d %Y", "%d %b %Y", "%d %B %Y",
        "%Y%m%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(val, fmt).strftime(TARGET_DATE_FMT)
        except ValueError:
            continue
    return None


def _is_normalized_date(val) -> bool:
    if pd.isna(val):
        return True
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(val)))


# --------------------------------------------------------------------------- #
#  Data generation                                                             #
# --------------------------------------------------------------------------- #

_MESSY_DATES = [
    "Jan 5 2024", "15/03/2023", "2022-07-20", "December 1 2023",
    "04-11-2021", "2023/08/30", "20230115", "Feb 28 2022",
    "2024-01-01", "March 15 2023",
]

_MESSY_PHONES = [
    "9876543210", "+91-98765-43210", "98765 43210", "(987) 654-3210",
    "9876-543210", "+919876543210", "987-654-3210", "9876543210",
    "+91 98765 43210", "9876543210",
]


def generate_dirty_data() -> Dict[str, pd.DataFrame]:
    np.random.seed(7)
    n = 20

    names  = [f"User_{i}" for i in range(1, n + 1)]
    emails = [f"user{i}@example.com" for i in range(1, n + 1)]

    rows = []
    for i in range(n):
        rows.append({
            "id":         i + 1,
            "name":       names[i],
            "email":      emails[i],
            "join_date":  _MESSY_DATES[i % len(_MESSY_DATES)],
            "phone":      _MESSY_PHONES[i % len(_MESSY_PHONES)],
            "score":      round(np.random.uniform(0, 100), 1),
        })

    # Inject exact duplicates (rows 3 and 4 are identical to row 1)
    rows.append({**rows[0], "id": n + 1})
    rows.append({**rows[0], "id": n + 2})

    # Inject near-duplicate: same person, email capitalised differently
    near_dup = {**rows[1], "id": n + 3, "email": rows[1]["email"].upper()}
    rows.append(near_dup)

    df = pd.DataFrame(rows)
    return {"main": df}


def get_issues(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    df = data["main"]

    # Exact duplicates (ignore id column)
    dup_mask = df.drop(columns=["id"]).duplicated(keep="first")
    exact_dup_indices = df.index[dup_mask].tolist()

    # Near-duplicates by normalised email
    df["_norm_email"] = df["email"].str.lower()
    near_dup_indices = df.index[df["_norm_email"].duplicated(keep="first")].tolist()
    df.drop(columns=["_norm_email"], inplace=True)

    # Bad dates
    bad_date_indices = df.index[~df["join_date"].apply(_is_normalized_date)].tolist()

    # Bad phones
    bad_phone_indices = df.index[~df["phone"].apply(_is_normalized_phone)].tolist()

    return {
        "initial_exact_dups":  len(exact_dup_indices),
        "initial_near_dups":   len(near_dup_indices),
        "initial_bad_dates":   len(bad_date_indices),
        "initial_bad_phones":  len(bad_phone_indices),
        "total_initial_rows":  len(df),
    }


# --------------------------------------------------------------------------- #
#  Action execution                                                            #
# --------------------------------------------------------------------------- #

def execute_action(action, data: Dict[str, pd.DataFrame]) -> Tuple[str, bool]:
    df = data.get("main")
    if df is None:
        return "Table 'main' not found.", False

    atype = action.action_type.value

    # ── inspect ──────────────────────────────────────────────────────────────
    if atype == "inspect":
        col = action.column
        if col and col in df.columns:
            sample = df[col].head(8).tolist()
            uniq   = df[col].nunique()
            return f"Column '{col}': {uniq} unique values. Sample: {sample}", True
        summary = {
            "rows":               len(df),
            "exact_duplicates":   int(df.drop(columns=["id"]).duplicated().sum()),
            "bad_date_count":     int((~df["join_date"].apply(_is_normalized_date)).sum()),
            "bad_phone_count":    int((~df["phone"].apply(_is_normalized_phone)).sum()),
        }
        return f"Table summary: {summary}", True

    # ── deduplicate ───────────────────────────────────────────────────────────
    if atype == "deduplicate":
        strategy = (action.strategy or "exact").lower()
        before = len(df)
        if strategy == "exact":
            df = df.drop_duplicates(subset=df.columns.difference(["id"]), keep="first")
        elif strategy == "fuzzy":
            df["_norm_email"] = df["email"].str.lower()
            df = df.drop_duplicates(subset=["_norm_email"], keep="first")
            df.drop(columns=["_norm_email"], inplace=True)
        else:
            return f"Unknown dedup strategy '{strategy}'. Use 'exact' or 'fuzzy'.", False
        data["main"] = df.reset_index(drop=True)
        removed = before - len(data["main"])
        return f"Removed {removed} duplicate rows using strategy='{strategy}'.", True

    # ── merge_rows ────────────────────────────────────────────────────────────
    if atype == "merge_rows":
        id1, id2 = action.id1, action.id2
        if id1 is None or id2 is None:
            return "merge_rows requires id1 and id2.", False
        mask1 = df["id"] == id1
        mask2 = df["id"] == id2
        if not mask1.any() or not mask2.any():
            return f"Row id {id1} or {id2} not found.", False
        # Keep id1 row, drop id2
        df = df[~mask2].reset_index(drop=True)
        data["main"] = df
        return f"Merged row {id2} into row {id1} (kept {id1}).", True

    # ── normalize_column ──────────────────────────────────────────────────────
    if atype == "normalize_column":
        col = action.column
        fmt = (action.format or "").upper()
        if col not in df.columns:
            return f"Column '{col}' not found.", False

        if col == "join_date" or fmt in ("YYYY-MM-DD", "DATE"):
            fixed, failed = 0, 0
            for idx, val in df[col].items():
                parsed = _parse_date(val)
                if parsed:
                    df.at[idx, col] = parsed
                    fixed += 1
                else:
                    failed += 1
            data["main"] = df
            return f"Normalized {fixed} dates in '{col}'; {failed} could not be parsed.", True

        if col == "phone" or fmt == "PHONE":
            df[col] = df[col].apply(_normalize_phone)
            data["main"] = df
            return f"Normalized phone numbers in '{col}' to 10-digit format.", True

        if fmt == "LOWERCASE":
            df[col] = df[col].str.lower()
            data["main"] = df
            return f"Lowercased column '{col}'.", True

        if fmt == "UPPERCASE":
            df[col] = df[col].str.upper()
            data["main"] = df
            return f"Uppercased column '{col}'.", True

        return f"Unrecognised format '{fmt}' for column '{col}'.", False

    # ── done ──────────────────────────────────────────────────────────────────
    if atype == "done":
        return "Agent signalled completion.", True

    return f"Action '{atype}' is not supported in Task 2.", False


# --------------------------------------------------------------------------- #
#  Grader                                                                      #
# --------------------------------------------------------------------------- #

def grade(data: Dict[str, pd.DataFrame], issues: Dict[str, Any]) -> float:
    df = data["main"]

    # ── dedup score (0.4 weight) ─────────────────────────────────────────────
    initial_dups = issues["initial_exact_dups"] + issues["initial_near_dups"]
    if initial_dups == 0:
        dedup_score = 1.0
    else:
        # Count remaining duplicates (exact + near)
        exact_remaining = int(df.drop(columns=["id"]).duplicated().sum())
        df["_ne"] = df["email"].str.lower()
        near_remaining = int(df["_ne"].duplicated().sum())
        df.drop(columns=["_ne"], inplace=True)
        remaining_dups = exact_remaining + near_remaining
        dedup_score = max(0.0, 1.0 - remaining_dups / initial_dups)

    # ── date score (0.3 weight) ──────────────────────────────────────────────
    n_rows = len(df)
    if n_rows == 0:
        date_score = 1.0
    else:
        good_dates = int(df["join_date"].apply(_is_normalized_date).sum())
        date_score = good_dates / n_rows

    # ── phone score (0.3 weight) ─────────────────────────────────────────────
    if n_rows == 0:
        phone_score = 1.0
    else:
        good_phones = int(df["phone"].apply(_is_normalized_phone).sum())
        phone_score = good_phones / n_rows

    total = round(0.4 * dedup_score + 0.3 * date_score + 0.3 * phone_score, 4)
    return total


def count_remaining_issues(data: Dict[str, pd.DataFrame]) -> int:
    df = data["main"]
    exact_dups  = int(df.drop(columns=["id"]).duplicated().sum())
    bad_dates   = int((~df["join_date"].apply(_is_normalized_date)).sum())
    bad_phones  = int((~df["phone"].apply(_is_normalized_phone)).sum())
    return exact_dups + bad_dates + bad_phones