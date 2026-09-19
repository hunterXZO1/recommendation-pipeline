from dataclasses import dataclass


@dataclass
class EvaluationResult:
    champion_metric: float
    challenger_metric: float
    improvement: float
    statistically_significant: bool


class ModelEvaluator:

    def evaluate(
        self,
        champion,
        challenger,
        evaluation_data,
    ) -> EvaluationResult:
        """
        Compare champion and challenger models.
        """
        raise NotImplementedError