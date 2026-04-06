"""
tests/test_env.py

Smoke tests — verify that all 3 tasks:
  1. reset() returns a valid observation
  2. inspect action works
  3. A known-good action sequence improves the score
  4. done terminates the episode

Run with:
    cd data-clean-env
    pip install -r requirements.txt
    python -m pytest tests/ -v
  OR simply:
    python tests/test_env.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CleaningAction, ActionType
from environment import DataCleanEnvironment


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def make_action(**kwargs) -> CleaningAction:
    return CleaningAction(**kwargs)


def _run_task_smoke(task_id: int, fix_actions: list[CleaningAction]):
    print(f"\n{'='*50}")
    print(f"  Task {task_id} smoke test")
    print(f"{'='*50}")

    env = DataCleanEnvironment(task_id=task_id)
    obs = env.reset()
    print(f"  reset()  → issues_remaining={obs.issues_remaining}  score={obs.partial_score}")
    assert obs.success,          "reset() should succeed"
    assert obs.issues_remaining > 0, "Should start with issues"
    assert not obs.done,         "Should not be done at start"

    # inspect
    inspect = make_action(action_type=ActionType.INSPECT)
    obs = env.step(inspect)
    print(f"  inspect  → {obs.message[:80]}")
    assert obs.success

    # Apply fix actions
    for action in fix_actions:
        obs = env.step(action)
        print(f"  {action.action_type.value:<22} → score={obs.partial_score:.3f}  issues={obs.issues_remaining}")

    # Signal done
    obs = env.step(make_action(action_type=ActionType.DONE))
    print(f"  done()   → score={obs.partial_score:.3f}  done={obs.done}")
    assert obs.done, "done action should terminate episode"

    return obs.partial_score


# --------------------------------------------------------------------------- #
#  Task 1                                                                      #
# --------------------------------------------------------------------------- #

def test_task1():
    fix_actions = [
        make_action(action_type=ActionType.FILL_NULLS, column="age",        value=0),
        make_action(action_type=ActionType.FILL_NULLS, column="salary",     value=0.0),
        make_action(action_type=ActionType.FILL_NULLS, column="department", value="Unknown"),
        make_action(action_type=ActionType.FILL_NULLS, column="city",       value="Unknown"),
        make_action(action_type=ActionType.FILL_NULLS, column="score",      value=0.0),
    ]
    score = _run_task_smoke(1, fix_actions)
    assert score >= 0.9, f"Task 1 should score ≥ 0.9 after filling all nulls, got {score}"
    print(f"  ✅ Task 1 PASSED  (score={score:.3f})")


# --------------------------------------------------------------------------- #
#  Task 2                                                                      #
# --------------------------------------------------------------------------- #

def test_task2():
    fix_actions = [
        make_action(action_type=ActionType.DEDUPLICATE,       strategy="exact"),
        make_action(action_type=ActionType.DEDUPLICATE,       strategy="fuzzy"),
        make_action(action_type=ActionType.NORMALIZE_COLUMN,  column="join_date", format="YYYY-MM-DD"),
        make_action(action_type=ActionType.NORMALIZE_COLUMN,  column="phone",     format="PHONE"),
    ]
    score = _run_task_smoke(2, fix_actions)
    assert score >= 0.8, f"Task 2 should score ≥ 0.8 after dedup+normalize, got {score}"
    print(f"  ✅ Task 2 PASSED  (score={score:.3f})")


# --------------------------------------------------------------------------- #
#  Task 3                                                                      #
# --------------------------------------------------------------------------- #

def test_task3():
    fix_actions = [
        make_action(action_type=ActionType.JOIN_INSPECT),
        make_action(action_type=ActionType.CAST_COLUMN,      column="age",      table="customers", target_type="int"),
        make_action(action_type=ActionType.CAST_COLUMN,      column="price",    table="orders",    target_type="float"),
        make_action(action_type=ActionType.REMOVE_ORPHANS,   column="customer_id"),
        make_action(action_type=ActionType.FLAG_OUTLIERS,    column="age",      table="customers"),
        make_action(action_type=ActionType.FIX_OUTLIER,      column="age",      table="customers", row_id=2, value=30),
        make_action(action_type=ActionType.FIX_OUTLIER,      column="age",      table="customers", row_id=4, value=25),
        make_action(action_type=ActionType.FIX_OUTLIER,      column="age",      table="customers", row_id=8, value=50),
        make_action(action_type=ActionType.FIX_OUTLIER,      column="balance",  table="customers", row_id=3, value=500.0),
        make_action(action_type=ActionType.FIX_OUTLIER,      column="balance",  table="customers", row_id=8, value=400.0),
        make_action(action_type=ActionType.NORMALIZE_COLUMN, column="currency", table="customers", format="USD"),
    ]
    score = _run_task_smoke(3, fix_actions)
    assert score >= 0.5, f"Task 3 should score ≥ 0.5 after fixes, got {score}"
    print(f"  ✅ Task 3 PASSED  (score={score:.3f})")


# --------------------------------------------------------------------------- #
#  Loop penalty test                                                           #
# --------------------------------------------------------------------------- #

def test_loop_penalty():
    print(f"\n{'='*50}")
    print(f"  Loop penalty test")
    print(f"{'='*50}")
    env = DataCleanEnvironment(task_id=1)
    env.reset()

    inspect = make_action(action_type=ActionType.INSPECT)
    scores  = []
    for _ in range(6):
        obs = env.step(inspect)
        scores.append(obs.partial_score)

    print(f"  Scores after repeated inspect: {scores}")
    # After 3+ consecutive identical actions, penalty should apply
    # Score won't rise but may drop or stay at 0 due to penalty on 0 base
    print(f"  ✅ Loop penalty test PASSED (no crash)")


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    test_task1()
    test_task2()
    test_task3()
    test_loop_penalty()
    print(f"\n{'='*50}")
    print("  ALL TESTS PASSED ✅")
    print(f"{'='*50}\n")