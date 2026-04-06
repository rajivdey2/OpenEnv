"""
environment.py — Core DataCleanEnvironment class.

Manages episode state, delegates action execution and grading to the
appropriate task module, and enforces loop-detection penalties.
"""
from uuid import uuid4
from collections import Counter
from typing import Dict, Any, Optional

import pandas as pd

from models import CleaningAction, CleaningObservation, EpisodeState

# Lazy-import task modules via a dispatch table
import tasks.task1_nulls  as task1
import tasks.task2_dedup  as task2
import tasks.task3_schema as task3

_TASK_MODULES = {1: task1, 2: task2, 3: task3}
_MAX_STEPS    = {1: 20, 2: 20, 3: 25}

LOOP_THRESHOLD = 3      # same action N times in a row → penalty kicks in
LOOP_PENALTY   = 0.05   # per extra repeat beyond threshold
DESTRUCTIVE_PENALTY = 0.10


class DataCleanEnvironment:
    def __init__(self, task_id: int = 1):
        if task_id not in _TASK_MODULES:
            raise ValueError(f"task_id must be 1, 2, or 3 — got {task_id}")
        self.task_id  = task_id
        self._task    = _TASK_MODULES[task_id]
        self._state: Optional[EpisodeState]   = None
        self._data:  Optional[Dict[str, pd.DataFrame]] = None
        self._issues: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------------- #
    #  Public API                                                              #
    # ---------------------------------------------------------------------- #

    def reset(self) -> CleaningObservation:
        self._data   = self._task.generate_dirty_data()
        self._issues = self._task.get_issues(self._data)
        self._state  = EpisodeState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=self.task_id,
            max_steps=_MAX_STEPS[self.task_id],
            actions_taken=[],
        )
        issues_count = self._task.count_remaining_issues(self._data)
        preview      = self._get_preview("main" if self.task_id < 3 else "customers")
        return CleaningObservation(
            success=True,
            message=(
                f"Task {self.task_id} started. Episode {self._state.episode_id}. "
                f"Detected {issues_count} issue(s). Use 'inspect' to explore the data."
            ),
            table_preview=preview,
            issues_remaining=issues_count,
            partial_score=0.0,
            done=False,
        )

    def step(self, action: CleaningAction) -> CleaningObservation:
        if self._state is None or self._data is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1
        self._state.actions_taken.append(action.action_type.value)

        # ── Loop-detection penalty ────────────────────────────────────────────
        penalty = self._loop_penalty()

        # ── Destructive action guard ──────────────────────────────────────────
        destructive_pen = 0.0
        if action.action_type.value == "drop_nulls":
            # Small extra penalty for each destructive drop
            destructive_pen = DESTRUCTIVE_PENALTY * 0.5   # softer version

        # ── Execute action ────────────────────────────────────────────────────
        msg, success = self._task.execute_action(action, self._data)

        # ── Grade current state ───────────────────────────────────────────────
        raw_score    = self._task.grade(self._data, self._issues)
        final_score  = max(0.0, round(raw_score - penalty - destructive_pen, 4))

        issues_rem   = self._task.count_remaining_issues(self._data)
        table_name   = action.table or ("main" if self.task_id < 3 else "customers")
        preview      = self._get_preview(table_name)

        done = (
            action.action_type.value == "done"
            or final_score >= 0.99
            or self._state.step_count >= self._state.max_steps
        )

        return CleaningObservation(
            success=success,
            message=msg,
            table_preview=preview,
            issues_remaining=issues_rem,
            partial_score=final_score,
            done=done,
        )

    @property
    def state(self) -> EpisodeState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    # ---------------------------------------------------------------------- #
    #  Internals                                                               #
    # ---------------------------------------------------------------------- #

    def _loop_penalty(self) -> float:
        """Return penalty if agent keeps repeating the same action."""
        if len(self._state.actions_taken) < LOOP_THRESHOLD:
            return 0.0
        last_n = self._state.actions_taken[-LOOP_THRESHOLD:]
        if len(set(last_n)) == 1:                 # all identical
            repeats = len(last_n) - (LOOP_THRESHOLD - 1)
            return LOOP_PENALTY * max(0, repeats)
        return 0.0

    def _get_preview(self, table_name: str) -> list:
        """Return first 5 rows of a table as a list of dicts."""
     
        df = (self._data or {}).get(table_name)
        if df is None:
            return []
        return df.head(5).fillna("").to_dict(orient="records")