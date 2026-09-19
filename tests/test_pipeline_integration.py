import pandas as pd

from src.pipeline import RecommendationPipeline
from src.drift.detector import KSNumericalDriftDetector
from src.evaluation.evaluator import EvaluationResult


class MockTrainer:
    def __init__(self):
        self.train_called = False

    def train(self, data):
        self.train_called = True
        return "challenger-model"


class MockEvaluator:
    def evaluate(self, champion, challenger, evaluation_data):
        return EvaluationResult(
            champion_metric=0.70,
            challenger_metric=0.75,
            improvement=0.05,
            statistically_significant=True,
        )


def test_real_drift_detector_triggers_pipeline():

    reference_data = pd.DataFrame({
        "age": [
            20, 21, 22, 23, 24,
            25, 26, 27, 28, 29,
        ]
    })

    current_data = pd.DataFrame({
        "age": [
            80, 81, 82, 83, 84,
            85, 86, 87, 88, 89,
        ]
    })

    evaluation_data = current_data.copy()

    trainer = MockTrainer()

    pipeline = RecommendationPipeline(
        drift_detector=KSNumericalDriftDetector(),
        model_trainer=trainer,
        evaluator=MockEvaluator(),
    )

    result = pipeline.run(
        reference_data=reference_data,
        current_data=current_data,
        evaluation_data=evaluation_data,
        champion_model="champion-model",
    )

    assert result.drift_result.drift_detected is True
    assert result.retraining_triggered is True
    assert trainer.train_called is True
    assert result.evaluation_result is not None
    assert result.model_promoted is True