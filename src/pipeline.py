from dataclasses import dataclass

import pandas as pd

from src.drift.detector import DriftDetector, DriftResult
from src.models.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator, EvaluationResult


@dataclass
class PipelineResult:
    drift_result: DriftResult
    retraining_triggered: bool
    evaluation_result: EvaluationResult | None
    model_promoted: bool


class RecommendationPipeline:
    """
    Coordinates the recommendation model lifecycle.

    Flow:
        Detect drift
        -> Retrain challenger if drift is detected
        -> Evaluate challenger
        -> Promote if statistically significant improvement exists
    """

    def __init__(
        self,
        drift_detector: DriftDetector,
        model_trainer: ModelTrainer,
        evaluator: ModelEvaluator,
    ):
        self.drift_detector = drift_detector
        self.model_trainer = model_trainer
        self.evaluator = evaluator

    def run(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        evaluation_data: pd.DataFrame,
        champion_model,
    ) -> PipelineResult:

        drift_result = self.drift_detector.detect(
            reference_data=reference_data,
            current_data=current_data,
        )

        if not drift_result.drift_detected:
            return PipelineResult(
                drift_result=drift_result,
                retraining_triggered=False,
                evaluation_result=None,
                model_promoted=False,
            )

        challenger_model = self.model_trainer.train(current_data)

        evaluation_result = self.evaluator.evaluate(
            champion=champion_model,
            challenger=challenger_model,
            evaluation_data=evaluation_data,
        )

        model_promoted = evaluation_result.statistically_significant

        return PipelineResult(
            drift_result=drift_result,
            retraining_triggered=True,
            evaluation_result=evaluation_result,
            model_promoted=model_promoted,
        )