from src.drift.detector import DriftResult
from src.evaluation.evaluator import EvaluationResult


def test_drift_result():
    result = DriftResult(
        drift_detected=True,
        drift_score=0.8,
        method="test",
    )

    assert result.drift_detected is True
    assert result.drift_score == 0.8


def test_evaluation_result():
    result = EvaluationResult(
        champion_metric=0.70,
        challenger_metric=0.75,
        improvement=0.05,
        statistically_significant=True,
    )

    assert result.challenger_metric > result.champion_metric
    assert result.statistically_significant is True