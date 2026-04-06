"""
Task 3: Schema Alignment + Type Correction + Referential Integrity (Hard)

Two related tables:
  - customers (id, name, age, country, currency, balance)
  - orders    (order_id, customer_id, product, price, quantity)

Issues injected:
  - age stored as float, should be int; outlier ages (999, -5)
  - price stored as string ("$12.50"), should be float
  - orphaned orders (customer_id references non-existent customers)
  - mixed currencies (USD, INR, EUR in same column)
  - balance outliers

Score = 0.25 × type_correctness
      + 0.25 × referential_integrity
      + 0.25 × outlier_handling
      + 0.25 × currency_consistency
"""
import re
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any


# --------------------------------------------------------------------------- #
#  Data generation                                                             #
# --------------------------------------------------------------------------- #

def generate_dirty_data() -> Dict[str, pd.DataFrame]:
    np.random.seed(99)

    # ── customers ─────────────────────────────────────────────────────────────
    customers = pd.DataFrame({
        "id":       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "name":     [f"Customer_{i}" for i in range(1, 11)],
        "age":      [25.0, 999.0, 32.0, -5.0, 45.0,   # outliers at index 1,3
                     28.0, 52.0, 150.0, 38.0, 41.0],  # outlier at index 7
        "country":  ["IN", "US", "US", "IN", "UK",
                     "IN", "US", "UK", "IN", "US"],
        "currency": ["INR", "USD", "USD", "EUR",  "GBP",    # mixed
                     "INR", "USD", "GBP", "INR",  "USD"],
        "balance":  [1500.0, 200.0, 99999.0, 50.0, 320.0,   # 99999 is outlier
                     180.0, 430.0, 9999.0, 270.0, 510.0],
    })

    # ── orders ────────────────────────────────────────────────────────────────
    orders = pd.DataFrame({
        "order_id":    list(range(101, 121)),
        "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,   # valid
                        11, 12, 99, 88, 3, 5, 7, 2, 1, 4], # 11,12,99,88 are orphans
        "product":     [f"Prod_{chr(65 + i % 10)}" for i in range(20)],
        "price":       ["$12.50", "$200.00", "150", "$45.99", "Rs.300",   # mixed formats
                        "$89.00", "$23.50", "€55.00", "$17.00", "$62.50",
                        "$10.00", "$34.00", "$5.50",  "$120.00", "$75.00",
                        "$30.00", "$95.00", "$42.00", "$18.50",  "$200.00"],
        "quantity":    np.random.randint(1, 10, size=20).tolist(),
    })

    return {"customers": customers, "orders": orders}


def get_issues(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    customers = data["customers"]
    orders    = data["orders"]

    # Type issues
    age_wrong_type   = not pd.api.types.is_integer_dtype(customers["age"])
    price_wrong_type = not pd.api.types.is_float_dtype(orders["price"])

    # Outlier ages: outside 0–120
    age_outlier_idx = customers.index[
        (customers["age"] < 0) | (customers["age"] > 120)
    ].tolist()

    # Balance outlier: > 9000 (99999 and 9999)
    balance_outlier_idx = customers.index[customers["balance"] > 9000].tolist()

    # Orphaned orders
    valid_cust_ids  = set(customers["id"])
    orphan_idx      = orders.index[~orders["customer_id"].isin(valid_cust_ids)].tolist()

    # Mixed currencies (anything that isn't "USD")
    non_usd_idx     = customers.index[customers["currency"] != "USD"].tolist()

    return {
        "age_wrong_type":       age_wrong_type,
        "price_wrong_type":     price_wrong_type,
        "age_outlier_indices":  age_outlier_idx,
        "balance_outlier_idx":  balance_outlier_idx,
        "orphan_order_indices": orphan_idx,
        "non_usd_count":        len(non_usd_idx),
        "initial_orphan_count": len(orphan_idx),
        "initial_outlier_count": len(age_outlier_idx) + len(balance_outlier_idx),
    }


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

_CURRENCY_RATES_TO_USD = {
    "USD": 1.0, "INR": 0.012, "EUR": 1.08, "GBP": 1.27,
}

def _parse_price(val) -> float | None:
    """Strip currency symbols and parse to float."""
    if pd.isna(val):
        return None
    cleaned = re.sub(r"[^\d.]", "", str(val))
    try:
        return float(cleaned)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
#  Action execution                                                            #
# --------------------------------------------------------------------------- #

def execute_action(action, data: Dict[str, pd.DataFrame]) -> Tuple[str, bool]:
    atype      = action.action_type.value
    table_name = action.table or "customers"
    df         = data.get(table_name)

    # ── inspect ───────────────────────────────────────────────────────────────
    if atype == "inspect":
        if df is None:
            return f"Table '{table_name}' not found. Available: {list(data.keys())}", False
        col = action.column
        if col and col in df.columns:
            sample  = df[col].head(8).tolist()
            dtype   = str(df[col].dtype)
            nulls   = int(df[col].isna().sum())
            return f"Column '{col}' (dtype={dtype}): {nulls} nulls. Sample: {sample}", True
        info = {
            "rows":    len(df),
            "columns": list(df.columns),
            "dtypes":  df.dtypes.astype(str).to_dict(),
            "nulls":   df.isna().sum().to_dict(),
        }
        return f"Table '{table_name}': {info}", True

    # ── join_inspect ──────────────────────────────────────────────────────────
    if atype == "join_inspect":
        customers = data.get("customers")
        orders    = data.get("orders")
        if customers is None or orders is None:
            return "Both 'customers' and 'orders' tables required.", False
        valid_ids  = set(customers["id"])
        orphans    = orders[~orders["customer_id"].isin(valid_ids)]
        return (
            f"Referential integrity check: {len(orphans)} orphaned orders "
            f"(customer_ids: {orphans['customer_id'].tolist()})",
            True,
        )

    # ── cast_column ───────────────────────────────────────────────────────────
    if atype == "cast_column":
        if df is None:
            return f"Table '{table_name}' not found.", False
        col         = action.column
        target_type = (action.target_type or "").lower()
        if col not in df.columns:
            return f"Column '{col}' not found in '{table_name}'.", False

        try:
            if target_type in ("int", "integer"):
                if col == "price":
                    df[col] = df[col].apply(_parse_price)
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            elif target_type in ("float", "double"):
                if col == "price":
                    df[col] = df[col].apply(_parse_price)
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            elif target_type in ("str", "string"):
                df[col] = df[col].astype(str)
            else:
                return f"Unknown target_type '{target_type}'.", False
            data[table_name] = df
            return f"Cast column '{col}' in '{table_name}' to {target_type}.", True
        except Exception as e:
            return f"Cast failed: {e}", False

    # ── remove_orphans ────────────────────────────────────────────────────────
    if atype == "remove_orphans":
        orders    = data.get("orders")
        customers = data.get("customers")
        if orders is None or customers is None:
            return "Both 'customers' and 'orders' tables are required.", False
        fk_col    = action.column or "customer_id"
        valid_ids = set(customers["id"])
        before    = len(orders)
        orders    = orders[orders[fk_col].isin(valid_ids)].reset_index(drop=True)
        data["orders"] = orders
        removed   = before - len(orders)
        return f"Removed {removed} orphaned orders (invalid {fk_col}).", True

    # ── flag_outliers ─────────────────────────────────────────────────────────
    if atype == "flag_outliers":
        if df is None:
            return f"Table '{table_name}' not found.", False
        col  = action.column
        rule = action.rule or ""
        if col not in df.columns:
            return f"Column '{col}' not found in '{table_name}'.", False

        if col == "age":
            outliers = df[(df[col] < 0) | (df[col] > 120)]
        elif col == "balance":
            outliers = df[df[col] > 9000]
        else:
            return f"No outlier rule defined for column '{col}'.", False

        ids = outliers["id"].tolist() if "id" in outliers.columns else outliers.index.tolist()
        return f"Flagged {len(outliers)} outliers in '{col}': row ids {ids}", True

    # ── fix_outlier ───────────────────────────────────────────────────────────
    if atype == "fix_outlier":
        if df is None:
            return f"Table '{table_name}' not found.", False
        col    = action.column
        row_id = action.row_id
        val    = action.value
        if col is None or row_id is None:
            return "fix_outlier requires column and row_id.", False
        id_col = "id" if "id" in df.columns else df.index
        mask   = df["id"] == row_id if "id" in df.columns else df.index == row_id
        if not mask.any():
            return f"Row with id={row_id} not found in '{table_name}'.", False
        df.loc[mask, col] = val
        data[table_name]  = df
        return f"Fixed outlier: set '{col}'={val} for row id={row_id}.", True

    # ── normalize_column (currency) ───────────────────────────────────────────
    if atype == "normalize_column":
        if df is None:
            return f"Table '{table_name}' not found.", False
        col = action.column
        fmt = (action.format or "").upper()
        if col == "currency" or fmt == "USD":
            converted = 0
            if "balance" in df.columns and "currency" in df.columns:
                for idx, row in df.iterrows():
                    rate = _CURRENCY_RATES_TO_USD.get(str(row["currency"]), 1.0)
                    df.at[idx, "balance"]  = round(row["balance"] * rate, 2)
                    df.at[idx, "currency"] = "USD"
                    converted += 1
                data[table_name] = df
                return f"Converted {converted} rows to USD.", True
            return "No 'balance' or 'currency' column found.", False
        return f"Unsupported normalization for column '{col}'.", False

    # ── done ──────────────────────────────────────────────────────────────────
    if atype == "done":
        return "Agent signalled completion.", True

    return f"Action '{atype}' is not supported in Task 3.", False


# --------------------------------------------------------------------------- #
#  Grader                                                                      #
# --------------------------------------------------------------------------- #

def grade(data: Dict[str, pd.DataFrame], issues: Dict[str, Any]) -> float:
    customers = data.get("customers", pd.DataFrame())
    orders    = data.get("orders",    pd.DataFrame())

    # ── 1. Type correctness (0.25) ────────────────────────────────────────────
    age_ok   = pd.api.types.is_integer_dtype(customers.get("age", pd.Series(dtype=float)))
    price_ok = pd.api.types.is_float_dtype(orders.get("price", pd.Series(dtype=object)))
    type_score = (0.5 * int(age_ok) + 0.5 * int(price_ok))

    # ── 2. Referential integrity (0.25) ───────────────────────────────────────
    initial_orphans = issues.get("initial_orphan_count", 1)
    if initial_orphans == 0:
        ri_score = 1.0
    else:
        valid_ids       = set(customers["id"]) if "id" in customers.columns else set()
        remaining_orphans = int((~orders["customer_id"].isin(valid_ids)).sum()) \
                           if "customer_id" in orders.columns else 0
        ri_score = max(0.0, 1.0 - remaining_orphans / initial_orphans)

    # ── 3. Outlier handling (0.25) ────────────────────────────────────────────
    initial_outliers = issues.get("initial_outlier_count", 1)
    if initial_outliers == 0:
        outlier_score = 1.0
    else:
        remaining = 0
        if "age" in customers.columns:
            remaining += int(((customers["age"] < 0) | (customers["age"] > 120)).sum())
        if "balance" in customers.columns:
            remaining += int((customers["balance"] > 9000).sum())
        outlier_score = max(0.0, 1.0 - remaining / initial_outliers)

    # ── 4. Currency consistency (0.25) ────────────────────────────────────────
    if "currency" not in customers.columns or len(customers) == 0:
        currency_score = 1.0
    else:
        usd_count      = int((customers["currency"] == "USD").sum())
        currency_score = usd_count / len(customers)

    total = round(
        0.25 * type_score
        + 0.25 * ri_score
        + 0.25 * outlier_score
        + 0.25 * currency_score,
        4,
    )
    return total


def count_remaining_issues(data: Dict[str, pd.DataFrame]) -> int:
    customers = data.get("customers", pd.DataFrame())
    orders    = data.get("orders",    pd.DataFrame())

    count = 0
    if "age" in customers.columns:
        count += int(((customers["age"] < 0) | (customers["age"] > 120)).sum())
    if "balance" in customers.columns:
        count += int((customers["balance"] > 9000).sum())
    if "customer_id" in orders.columns and "id" in customers.columns:
        valid_ids = set(customers["id"])
        count += int((~orders["customer_id"].isin(valid_ids)).sum())
    if "currency" in customers.columns:
        count += int((customers["currency"] != "USD").sum())
    return count