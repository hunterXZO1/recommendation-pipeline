from dataclasses import dataclass
import pandas as pd


@dataclass
class DriftResult:
    drift_detected: bool
    drift_score: float
    method: str


class DriftDetector:
    def detect(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> DriftResult:
        """
        Compare reference and current interaction data.

        This method will later contain the actual
        statistical drift detection logic.
        """
        raise NotImplementedError