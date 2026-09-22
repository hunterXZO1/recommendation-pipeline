"""Generates synthetic quiz interaction data.

Produces realistic-looking interaction records so the drift detector and
the recommendation model have data to run against before real usage data
exists. Correctness rate is deliberately drifted over the time window so
drift detection has something genuine to detect (rather than pure noise).
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import pandas as pd

from .quiz_core import Question
from .interaction_events import (
    INTERACTION_COLUMNS,
    append_interactions,
    compute_reward,
)

CATEGORIES = ["science", "history", "geography", "sports", "movies", "music"]
DIFFICULTIES = ["easy", "medium", "hard"]
OPTIONS = ["A", "B", "C", "D"]


def _make_question_bank(n_questions: int, seed: int) -> list[Question]:
    rng = random.Random(seed)
    questions = []
    for qid in range(1, n_questions + 1):
        options = OPTIONS.copy()
        correct = rng.choice(options)
        questions.append(
            Question(
                question_id=qid,
                category=rng.choice(CATEGORIES),
                difficulty=rng.choice(DIFFICULTIES),
                text=f"Sample question #{qid}",
                options=options,
                correct_answer=correct,
            )
        )
    return questions


def generate_synthetic_dataset(
    n_users: int = 200,
    n_questions: int = 60,
    n_interactions: int = 4000,
    start_timestamp: float | None = None,
    span_seconds: float = 30 * 24 * 3600,  # 30 days
    drift_at_fraction: float = 0.6,
    base_accuracy: float = 0.65,
    drifted_accuracy: float = 0.40,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic interaction DataFrame matching the schema.

    Simulates a population whose accuracy drops partway through the time
    window (e.g. harder question mix rolled out, or user base shifted) so
    the KS drift detector has a genuine distributional shift to catch, in
    addition to a "before" region for the reference distribution.

    Args:
        n_users: number of distinct synthetic users.
        n_questions: size of the question bank to draw from.
        n_interactions: total number of interaction rows to generate.
        start_timestamp: unix time for the first interaction; defaults to
            `span_seconds` before now.
        span_seconds: total time window the interactions are spread over.
        drift_at_fraction: point (0-1) in the time window where accuracy
            shifts from `base_accuracy` to `drifted_accuracy`.
        base_accuracy: P(correct) before the drift point.
        drifted_accuracy: P(correct) after the drift point.
        seed: RNG seed, for reproducibility.

    Returns:
        A pandas DataFrame with columns == INTERACTION_COLUMNS, sorted by
        timestamp.
    """
    rng = random.Random(seed)
    questions = _make_question_bank(n_questions, seed=seed)
    questions_by_id = {q.question_id: q for q in questions}

    if start_timestamp is None:
        start_timestamp = time.time() - span_seconds

    attempt_counts: dict[tuple[int, int], int] = {}  # (user_id, question_id) -> count
    rows = []

    for i in range(n_interactions):
        frac = i / max(n_interactions - 1, 1)
        ts = start_timestamp + frac * span_seconds

        user_id = rng.randint(1, n_users)
        question = rng.choice(questions)
        session_id = f"sess-{user_id}-{int(ts // 3600)}"  # coarse session grouping

        accuracy = drifted_accuracy if frac >= drift_at_fraction else base_accuracy
        is_correct = rng.random() < accuracy
        selected = (
            question.correct_answer
            if is_correct
            else rng.choice([o for o in question.options if o != question.correct_answer])
        )

        key = (user_id, question.question_id)
        attempt_counts[key] = attempt_counts.get(key, 0) + 1

        rows.append(
            {
                "user_id": user_id,
                "quiz_session_id": session_id,
                "question_id": question.question_id,
                "category": question.category,
                "difficulty": question.difficulty,
                "selected_answer": selected,
                "correct_answer": question.correct_answer,
                "is_correct": bool(selected == question.correct_answer),
                "reward": compute_reward(selected == question.correct_answer),
                "attempt_number": attempt_counts[key],
                "timestamp": float(ts),
            }
        )

    df = pd.DataFrame(rows, columns=INTERACTION_COLUMNS)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def write_sample_dataset(path: str | Path = "data/sample_interactions.csv", **kwargs) -> Path:
    """Generate a synthetic dataset and write it fresh to `path` (overwrites)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    df = generate_synthetic_dataset(**kwargs)
    append_interactions(df.to_dict("records"), path)
    return path


if __name__ == "__main__":
    out_path = write_sample_dataset()
    print(f"Wrote synthetic dataset to {out_path}")
