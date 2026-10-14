# recommendation-pipeline

A Python machine-learning pipeline that trains a PyTorch two-tower recommender,
detects data drift, retrains a challenger model on recent interactions, evaluates
the champion against the challenger, and promotes the challenger atomically when
it is statistically better.

---

## Project layout

```
scripts/
  train_model.py          Trains and saves the initial champion checkpoint.
  run_pipeline.py         Runs the full pipeline end-to-end.
  retrain_recent.py       Standalone script: retrain on recent data only.
  test_recommendations.py Quick sanity check for the saved model.

src/
  pipeline.py             RecommendationPipeline orchestrator.
  data/
    movielens.py          Loads MovieLens ratings; get_recent_interactions.
    loader.py             Generic CSV loader for quiz interaction data.
  drift/
    detector.py           KSNumericalDriftDetector (KS two-sample test).
  evaluation/
    evaluator.py          ModelEvaluator: paired Hit Rate@K + bootstrap test.
  models/
    two_tower.py          TwoTowerRecommender (PyTorch nn.Module).
    pytorch_trainer.py    PyTorchModelTrainer: train / save / load / recommend.
    trainer.py            Abstract ModelTrainer interface.
  quiz/
    quiz_core.py          Question, QuizSession - quiz delivery and scoring.
    interaction_events.py Schema, record_interaction, append_interactions.
    data_generator.py     Synthetic quiz data generator for testing.

tests/                    Automated test suite (pytest).
data/movielens/           Place ratings.dat here before running scripts.
artifacts/                Saved model checkpoints (.pt files).
docs/                     Schema and module documentation.
```

---

## Quick start

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Prepare data

Download MovieLens 1M and place `ratings.dat` at `data/movielens/ratings.dat`.

### 3. Train the initial champion

```
python -m scripts.train_model
```

This trains on the **oldest 80%** of interactions (chronologically sorted) and
saves the model to `artifacts/challenger.pt`.  Training on the pre-evaluation
window prevents data leakage into the champion checkpoint.

### 4. Run the pipeline

```
python -m scripts.run_pipeline
```

The pipeline:
1. Splits data into reference (oldest 80%) and current (newest 20%).
2. Splits current into training_current (80%) and evaluation (20%).
3. Detects drift between reference and current using the KS test.
4. If drift is detected, retrains a challenger on recent interactions from
   training_current.
5. Evaluates champion vs challenger on the held-out evaluation set using
   paired Hit Rate@K and a paired bootstrap significance test.
6. If the challenger is statistically better, atomically promotes it by
   overwriting the champion checkpoint.

### 5. Run the tests

```
python -m pytest tests/ -v
```

Expected: **68 tests pass**.

---

## Data leakage guarantee

The champion checkpoint (`artifacts/challenger.pt`) is trained in
`scripts/train_model.py` using only the **oldest 80% of all interactions**.
The pipeline uses the **newest 20%** for challenger training (recent window)
and evaluation (held-out window).  These windows never overlap.

Regression tests in `tests/test_leakage.py` verify this chronological
separation automatically on every test run.

---

## Model promotion

When the pipeline promotes the challenger, it:
1. Writes the new checkpoint to `<champion_path>.tmp` first.
2. Atomically renames the temp file over the champion path.

The old champion file is never deleted until the replacement is fully written.
If the write fails, the temp file is cleaned up and the original is preserved.

---

## Quiz data support

The pipeline accepts quiz interaction data in addition to MovieLens data.
Quiz interactions use `question_id` (mapped to `item_id` automatically) and
the same `user_id`, `reward`, `timestamp` schema.  See
`docs/interaction_schema.md` for the full column reference and
`docs/quiz_module.md` for the quiz module guide.

---

## Configuration

Key parameters in `scripts/run_pipeline.py`:

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `data/movielens/ratings.dat` | Interaction data file |
| `CHAMPION_PATH` | `artifacts/challenger.pt` | Champion checkpoint |
| `CHALLENGER_PATH` | `artifacts/challenger_recent.pt` | Challenger checkpoint |
| `recent_days` | `30` | Window for recent interactions used in challenger training |
| `top_k` | `10` | Hit Rate@K cutoff |
