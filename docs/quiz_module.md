# Quiz / User Interaction Module

Owner: Nav
Location: `src/quiz/`

This module owns quiz question delivery, answer handling, and turning
completed quiz answers into interaction records the rest of the
pipeline consumes. It has no dependency on model internals — see the
team guide's "do not couple quiz UI to model internals" rule.

## Files

| File | Purpose |
|---|---|
| `src/quiz/quiz_core.py` | `Question`, `QuizSession` — question delivery, answer capture, scoring |
| `src/quiz/interaction_events.py` | Schema, `record_interaction`, storage (`append_interaction(s)`, `read_interactions`), `validate_interaction_schema` |
| `src/quiz/data_generator.py` | `generate_synthetic_dataset`, `write_sample_dataset` — synthetic data for drift/training testing |
| `docs/interaction_schema.md` | The schema contract itself |
| `tests/test_quiz.py` | Unit + edge-case tests for all of the above |

## How to run the quiz (example)

```python
from quiz.quiz_core import Question, QuizSession
from quiz.interaction_events import record_interaction, append_interactions

questions = [
    Question(1, "science", "easy", "2+2?", ["3", "4", "5"], "4"),
    Question(2, "history", "medium", "Year WW2 ended?", ["1943", "1945", "1950"], "1945"),
]

session = QuizSession(user_id=101, questions=questions)
session.answer(question_id=1, selected_answer="4")
session.answer(question_id=2, selected_answer="1943")

answered = session.finish()
print("score:", session.score())  # 0.5

# Turn the completed quiz into interaction rows and store them —
# no manual conversion step needed.
rows = [record_interaction(a) for a in answered]
append_interactions(rows, "data/interactions.csv")
```

## Interaction event format

See `docs/interaction_schema.md` for the full column reference. Summary:

```text
user_id, quiz_session_id, question_id, category, difficulty,
selected_answer, correct_answer, is_correct, reward,
attempt_number, timestamp
```

One row is produced per question answered (not per full quiz attempt).

## Example input/output

**Input** (calling `session.answer(question_id=2, selected_answer="1943")`):

```python
AnsweredQuestion(
    user_id=101,
    quiz_session_id="sess-8f3a...",
    question=<Question id=2 "Year WW2 ended?">,
    selected_answer="1943",
    attempt_number=1,
    timestamp=1758540213.0,
    is_correct=False,
)
```

**Output** (after `record_interaction(...)`):

```python
{
    "user_id": 101,
    "quiz_session_id": "sess-8f3a...",
    "question_id": 2,
    "category": "history",
    "difficulty": "medium",
    "selected_answer": "1943",
    "correct_answer": "1945",
    "is_correct": False,
    "reward": 0.0,
    "attempt_number": 1,
    "timestamp": 1758540213.0,
}
```

## How the pipeline consumes the data

Interaction rows are appended to a CSV (default `data/interactions.csv`).
This is the file `load_interactions(path) -> DataFrame` (owned by
Kanishka's integration layer) reads from.

**Important:** always read this CSV with `read_interactions(path)` from
`interaction_events.py`, not a bare `pd.read_csv`. Some answer values
look numeric (e.g. `"1945"`), and without explicit dtypes pandas will
silently cast them to `int`, corrupting the schema on round-trip.
`load_interactions()` in `src/data/loader.py` should apply the same
`CSV_READ_DTYPES` mapping exported from this module — flagged for
Kanishka.

## Generating sample/test data

For drift detection and model training/testing before real usage data
exists:

```bash
python -m src.quiz.data_generator
# or:
python -c "from quiz.data_generator import write_sample_dataset; write_sample_dataset()"
```

This writes `data/sample_interactions.csv` with a simulated accuracy
drop partway through the time window, so the KS drift detector has a
genuine distributional shift to detect rather than pure noise.

## Running tests

```bash
python -m pytest tests/test_quiz.py -v
```

Covers: quiz answer flow and scoring, edge cases (empty quiz, invalid
answer, unknown question, answering after finish, repeated attempts),
interaction event generation, schema validation (missing columns,
nulls, dtype checks including the numeric-string round-trip gotcha
above), and the synthetic data generator (schema match, reproducibility,
and that the simulated drift is actually detectable).

## Open items before this is fully "done" per the team guide

- [ ] Confirm reward shape (binary vs. weighted) with Prince
- [ ] Confirm per-question granularity with Kanishka/Prince
- [ ] Kanishka: apply `CSV_READ_DTYPES` in `src/data/loader.py`
- [ ] PR opened from `feature/nav` (or `feature/quiz-platform`) into `main`
