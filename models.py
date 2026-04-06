from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum


class ActionType(str, Enum):
    FILL_NULLS = "fill_nulls"
    DROP_NULLS = "drop_nulls"
    DEDUPLICATE = "deduplicate"
    NORMALIZE_COLUMN = "normalize_column"
    CAST_COLUMN = "cast_column"
    REMOVE_ORPHANS = "remove_orphans"
    FLAG_OUTLIERS = "flag_outliers"
    FIX_OUTLIER = "fix_outlier"
    INSPECT = "inspect"
    MERGE_ROWS = "merge_rows"
    JOIN_INSPECT = "join_inspect"
    DONE = "done"


class CleaningAction(BaseModel):
    action_type: ActionType
    column: Optional[str] = None
    table: Optional[str] = None
    value: Optional[Any] = None
    strategy: Optional[str] = None      # e.g. "exact", "fuzzy"
    format: Optional[str] = None        # e.g. "YYYY-MM-DD"
    target_type: Optional[str] = None   # e.g. "int", "float", "str"
    row_id: Optional[int] = None
    rule: Optional[str] = None          # e.g. "age > 120 OR age < 0"
    ref_table: Optional[str] = None
    id1: Optional[int] = None
    id2: Optional[int] = None


class CleaningObservation(BaseModel):
    success: bool
    message: str                                    # human-readable result
    table_preview: Optional[List[Dict]] = None      # first 5 rows of affected table
    issues_remaining: int                           # how many known issues still exist
    partial_score: float                            # current reward so far
    done: bool


class CleaningReward(BaseModel):
    value: float                        # 0.0 - 1.0
    breakdown: Dict[str, float]         # per-criterion scores
    penalties: float                    # subtracted for bad behavior


class EpisodeState(BaseModel):
    episode_id: str
    step_count: int
    task_id: int                        # 1, 2, or 3
    max_steps: int
    actions_taken: List[str]