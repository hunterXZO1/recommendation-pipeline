# Quiz Interaction Data Schema

Owner: Nav (Quiz / User Interaction module)
Status: Draft — pending confirmation from Kanishka (integration) and Prince (ML)

## Purpose

This is the contract between the quiz module and the rest of the pipeline.
Every completed question produces exactly one row in this schema. The
recommendation pipeline consumes this data through
`load_interactions(path) -> DataFrame` (see `src/data/loader.py`), so any
change to these columns must be discussed with Kanishka before merging.

## Granularity

**One row per question answered**, not one row per full quiz attempt. This
gives far more interaction density for drift detection and model
training. A full quiz attempt can be reconstructed by grouping rows on
`user_id` + `quiz_session_id`.

## Columns

| Column | Type | Nullable | Description |
|---|---|---|---|
| `user_id` | int | No | Unique identifier for the user taking the quiz |
| `quiz_session_id` | str | No | Groups all questions answered in one quiz attempt |
| `question_id` | int | No | Unique identifier for the quiz question (MovieLens `movie_id` equivalent) |
| `category` | str | No | Quiz topic/category, e.g. `"science"`, `"history"` |
| `difficulty` | str | No | One of `"easy"`, `"medium"`, `"hard"` |
| `selected_answer` | str | No | The option the user selected |
| `correct_answer` | str | No | The ground-truth correct option |
| `is_correct` | bool | No | `selected_answer == correct_answer` |
| `reward` | float | No | Numeric training signal (MovieLens `rating` equivalent). `1.0` if correct, `0.0` if incorrect, in the current (binary) version |
| `attempt_number` | int | No | How many times this user has answered this exact `question_id` before (1 = first time) |
| `timestamp` | float | No | Unix timestamp (seconds) of when the answer was submitted |

## Example rows

```text
user_id, quiz_session_id, question_id, category, difficulty, selected_answer, correct_answer, is_correct, reward, attempt_number, timestamp
101, "sess-8f3a", 42, "science", "medium", "B", "B", True, 1.0, 1, 1758540213.0
101, "sess-8f3a", 57, "history", "hard",   "A", "C", False, 0.0, 1, 1758540240.0
102, "sess-1c9d", 42, "science", "medium", "C", "B", False, 0.0, 2, 1758540300.0
```

## Open design decisions (flagged for Kanishka / Prince)

1. **Reward shape** — currently binary (1.0 / 0.0). Could later be
   weighted by difficulty or response time. Defaulting to binary for
   Review II; revisit if the ML side wants a richer signal.
2. **Granularity** — per-question rows (decided above, not per full
   attempt) unless there's a strong objection.

If either of these needs to change, update this file first, then notify
both teammates — this schema is a shared interface per the team guide's
golden rule ("interfaces before implementation").

## Storage format

Rows are appended to a single CSV at `data/interactions.csv` (path is
configurable). This file is what `load_interactions(path)` reads. New
quiz completions append directly to this file — no manual conversion
step is required.
