"""Turns quiz results into pipeline-ready interaction records.

This is the single place that produces rows matching
`docs/interaction_schema.md`. A completed quiz answer becomes a usable
interaction record automatically — no manual conversion step required.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .quiz_core import AnsweredQuestion

# Canonical column order/schema. Keep this in sync with
# docs/interaction_schema.md and with src/data/loader.py's expectations.
INTERACTION_COLUMNS: list[str] = [
    "user_id",
    "quiz_session_id",
    "question_id",
    "category",
    "difficulty",
    "selected_answer",
    "correct_answer",
    "is_correct",
    "reward",
    "attempt_number",
    "timestamp",
]

# Expected pandas dtype "kind" per column (see numpy dtype.kind docs):
#   i = signed int, u = unsigned int, f = float, b = bool, O = object (str)
# dtype hints for reading the CSV back. Without this, pandas silently
# infers numeric-looking string answers (e.g. "1945") as int64, which
# corrupts the schema on round-trip. Anything reading this CSV — Nav's own
# code or Kanishka's load_interactions() — should read it with these dtypes.
CSV_READ_DTYPES: dict[str, str] = {
    "quiz_session_id": "str",
    "category": "str",
    "difficulty": "str",
    "selected_answer": "str",
    "correct_answer": "str",
}

_EXPECTED_DTYPE_KINDS: dict[str, str] = {
    "user_id": "iu",
    "quiz_session_id": "O",
    "question_id": "iu",
    "category": "O",
    "difficulty": "O",
    "selected_answer": "O",
    "correct_answer": "O",
    "is_correct": "b",
    "reward": "fiu",
    "attempt_number": "iu",
    "timestamp": "fiu",
}


def compute_reward(is_correct: bool) -> float:
    """Binary reward signal. Kept as its own function so the reward shape
    (e.g. difficulty-weighted, time-weighted) can be swapped later without
    touching call sites.
    """
    return 1.0 if is_correct else 0.0


def record_interaction(answered: AnsweredQuestion) -> dict:
    """Convert one AnsweredQuestion into a schema-matching dict row."""
    q = answered.question
    return {
        "user_id": answered.user_id,
        "quiz_session_id": answered.quiz_session_id,
        "question_id": q.question_id,
        "category": q.category,
        "difficulty": q.difficulty,
        "selected_answer": answered.selected_answer,
        "correct_answer": q.correct_answer,
        "is_correct": bool(answered.is_correct),
        "reward": compute_reward(answered.is_correct),
        "attempt_number": answered.attempt_number,
        "timestamp": float(answered.timestamp),
    }


def validate_interaction_schema(df: pd.DataFrame) -> None:
    """Raises ValueError if `df` doesn't match the interaction schema.

    Checks: required columns present, no unexpected nulls, and basic
    type sanity per column.
    """
    missing = [c for c in INTERACTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Interaction data missing required columns: {missing}")

    if df[INTERACTION_COLUMNS].isnull().any().any():
        bad_cols = df[INTERACTION_COLUMNS].columns[
            df[INTERACTION_COLUMNS].isnull().any()
        ].tolist()
        raise ValueError(f"Interaction data has nulls in columns: {bad_cols}")

    if len(df) == 0:
        return  # nothing further to type-check

    for col, allowed_kinds in _EXPECTED_DTYPE_KINDS.items():
        actual_kind = df[col].dtype.kind
        if actual_kind not in allowed_kinds:
            raise ValueError(
                f"Column {col!r} expected dtype kind in {list(allowed_kinds)}, "
                f"got dtype {df[col].dtype} (kind={actual_kind!r})"
            )


def append_interaction(row: dict, path: str | os.PathLike) -> None:
    """Append a single interaction row to the CSV at `path`.

    Creates the file (with header) if it doesn't exist yet. This is what
    lets new quiz interactions continuously enter the pipeline with zero
    manual steps.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_row = pd.DataFrame([row], columns=INTERACTION_COLUMNS)
    write_header = not path.exists() or path.stat().st_size == 0

    df_row.to_csv(path, mode="a", header=write_header, index=False)


def append_interactions(rows: list[dict], path: str | os.PathLike) -> None:
    """Bulk version of append_interaction — appends many rows at once."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_rows = pd.DataFrame(rows, columns=INTERACTION_COLUMNS)
    write_header = not path.exists() or path.stat().st_size == 0

    df_rows.to_csv(path, mode="a", header=write_header, index=False)


def read_interactions(path: str | os.PathLike) -> pd.DataFrame:
    """Read an interactions CSV back with the correct dtypes.

    Use this (not a bare `pd.read_csv`) anywhere the interaction data is
    consumed — it prevents pandas from silently inferring numeric-looking
    answer strings (e.g. "1945") as integers.
    """
    return pd.read_csv(path, dtype=CSV_READ_DTYPES)
