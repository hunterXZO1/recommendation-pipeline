"""Tests for the quiz / interaction module (Nav's component).

Covers: quiz answering flow, edge cases, interaction event generation,
schema validation, and the synthetic data generator.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quiz.quiz_core import Question, QuizSession, InvalidAnswerError
from quiz.interaction_events import (
    INTERACTION_COLUMNS,
    record_interaction,
    append_interaction,
    append_interactions,
    read_interactions,
    validate_interaction_schema,
    compute_reward,
)
from quiz.data_generator import generate_synthetic_dataset


# ---------- fixtures ----------

@pytest.fixture
def sample_questions():
    return [
        Question(1, "science", "easy", "2+2?", ["3", "4", "5"], "4"),
        Question(2, "history", "medium", "Year WW2 ended?", ["1943", "1945", "1950"], "1945"),
        Question(3, "geography", "hard", "Capital of Mongolia?", ["Ulaanbaatar", "Astana", "Bishkek"], "Ulaanbaatar"),
    ]


# ---------- Question construction ----------

def test_question_rejects_correct_answer_not_in_options():
    with pytest.raises(ValueError):
        Question(1, "science", "easy", "?", ["A", "B"], "C")


def test_question_rejects_invalid_difficulty():
    with pytest.raises(ValueError):
        Question(1, "science", "impossible", "?", ["A", "B"], "A")


# ---------- Quiz flow / answer handling ----------

def test_answering_correct_question(sample_questions):
    session = QuizSession(user_id=101, questions=sample_questions)
    result = session.answer(question_id=1, selected_answer="4")
    assert result.is_correct is True
    assert result.user_id == 101
    assert result.attempt_number == 1


def test_answering_incorrect_question(sample_questions):
    session = QuizSession(user_id=101, questions=sample_questions)
    result = session.answer(question_id=1, selected_answer="3")
    assert result.is_correct is False


def test_score_computed_correctly(sample_questions):
    session = QuizSession(user_id=101, questions=sample_questions)
    session.answer(1, "4")       # correct
    session.answer(2, "1943")    # incorrect
    session.answer(3, "Ulaanbaatar")  # correct
    assert session.score() == pytest.approx(2 / 3)


def test_finish_returns_all_answered(sample_questions):
    session = QuizSession(user_id=101, questions=sample_questions)
    session.answer(1, "4")
    session.answer(2, "1945")
    results = session.finish()
    assert len(results) == 2


def test_repeated_attempts_increment_attempt_number(sample_questions):
    session = QuizSession(user_id=101, questions=sample_questions)
    r1 = session.answer(1, "4")
    r2 = session.answer(1, "3")
    assert r1.attempt_number == 1
    assert r2.attempt_number == 2


# ---------- Edge cases ----------

def test_empty_quiz_rejected():
    with pytest.raises(ValueError):
        QuizSession(user_id=101, questions=[])


def test_invalid_answer_option_rejected(sample_questions):
    session = QuizSession(user_id=101, questions=sample_questions)
    with pytest.raises(InvalidAnswerError):
        session.answer(question_id=1, selected_answer="not-an-option")


def test_unknown_question_id_rejected(sample_questions):
    session = QuizSession(user_id=101, questions=sample_questions)
    with pytest.raises(KeyError):
        session.answer(question_id=999, selected_answer="4")


def test_cannot_answer_after_finish(sample_questions):
    session = QuizSession(user_id=101, questions=sample_questions)
    session.answer(1, "4")
    session.finish()
    with pytest.raises(RuntimeError):
        session.answer(2, "1945")


def test_score_with_no_answers_is_zero(sample_questions):
    session = QuizSession(user_id=101, questions=sample_questions)
    assert session.score() == 0.0


# ---------- Interaction event generation ----------

def test_record_interaction_matches_schema(sample_questions):
    session = QuizSession(user_id=101, questions=sample_questions)
    answered = session.answer(1, "4")
    row = record_interaction(answered)
    assert set(row.keys()) == set(INTERACTION_COLUMNS)
    assert row["is_correct"] is True
    assert row["reward"] == 1.0
    assert row["question_id"] == 1


def test_compute_reward_binary():
    assert compute_reward(True) == 1.0
    assert compute_reward(False) == 0.0


def test_append_interaction_creates_file_with_header(tmp_path, sample_questions):
    session = QuizSession(user_id=101, questions=sample_questions)
    answered = session.answer(1, "4")
    row = record_interaction(answered)

    out_path = tmp_path / "interactions.csv"
    append_interaction(row, out_path)

    df = read_interactions(out_path)
    assert list(df.columns) == INTERACTION_COLUMNS
    assert len(df) == 1


def test_append_interaction_appends_without_duplicating_header(tmp_path, sample_questions):
    session = QuizSession(user_id=101, questions=sample_questions)
    out_path = tmp_path / "interactions.csv"

    for qid, ans in [(1, "4"), (2, "1945"), (3, "Astana")]:
        answered = session.answer(qid, ans)
        append_interaction(record_interaction(answered), out_path)

    df = pd.read_csv(out_path)
    assert len(df) == 3
    # header should appear exactly once
    raw_text = out_path.read_text()
    assert raw_text.count("user_id") == 1


def test_completed_quiz_produces_usable_interaction_records_end_to_end(tmp_path, sample_questions):
    """A completed quiz attempt should produce interaction rows with no
    manual conversion step — this is the integration contract."""
    session = QuizSession(user_id=202, questions=sample_questions)
    session.answer(1, "4")
    session.answer(2, "1943")
    answered_questions = session.finish()

    rows = [record_interaction(a) for a in answered_questions]
    out_path = tmp_path / "interactions.csv"
    append_interactions(rows, out_path)

    df = read_interactions(out_path)
    validate_interaction_schema(df)
    assert len(df) == 2
    assert (df["user_id"] == 202).all()
    # numeric-looking answers ("1943") must survive round-trip as strings
    assert df["selected_answer"].dtype.kind == "O"


# ---------- Schema validation ----------

def test_validate_schema_rejects_missing_column():
    df = pd.DataFrame([{"user_id": 1}])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_interaction_schema(df)


def test_validate_schema_rejects_nulls(sample_questions):
    session = QuizSession(user_id=101, questions=sample_questions)
    answered = session.answer(1, "4")
    row = record_interaction(answered)
    row["category"] = None
    df = pd.DataFrame([row], columns=INTERACTION_COLUMNS)
    with pytest.raises(ValueError, match="nulls"):
        validate_interaction_schema(df)


def test_validate_schema_accepts_well_formed_data(sample_questions):
    session = QuizSession(user_id=101, questions=sample_questions)
    answered = session.answer(1, "4")
    row = record_interaction(answered)
    df = pd.DataFrame([row], columns=INTERACTION_COLUMNS)
    validate_interaction_schema(df)  # should not raise


# ---------- Synthetic data generator ----------

def test_generated_dataset_matches_schema():
    df = generate_synthetic_dataset(n_users=10, n_questions=5, n_interactions=50)
    assert list(df.columns) == INTERACTION_COLUMNS
    assert len(df) == 50
    validate_interaction_schema(df)


def test_generated_dataset_is_sorted_by_timestamp():
    df = generate_synthetic_dataset(n_users=10, n_questions=5, n_interactions=50)
    assert (df["timestamp"].diff().dropna() >= 0).all()


def test_generated_dataset_reproducible_with_seed():
    df1 = generate_synthetic_dataset(n_users=5, n_questions=5, n_interactions=20, seed=7)
    df2 = generate_synthetic_dataset(n_users=5, n_questions=5, n_interactions=20, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_generated_dataset_shows_accuracy_shift():
    """Sanity check that the drift simulation actually produces a
    detectable shift in correctness rate, for the drift detector to find."""
    df = generate_synthetic_dataset(
        n_users=50,
        n_questions=20,
        n_interactions=3000,
        base_accuracy=0.8,
        drifted_accuracy=0.3,
        drift_at_fraction=0.5,
        seed=1,
    )
    midpoint = len(df) // 2
    early_accuracy = df.iloc[:midpoint]["is_correct"].mean()
    late_accuracy = df.iloc[midpoint:]["is_correct"].mean()
    assert early_accuracy - late_accuracy > 0.2
