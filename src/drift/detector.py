from dataclasses import dataclass

import pandas as pd
from scipy.stats import ks_2samp


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

        This base class defines the interface for drift detectors.
        """
        raise NotImplementedError


class KSNumericalDriftDetector(DriftDetector):
    """
    Detects distribution drift in numerical features
    using the two-sample Kolmogorov-Smirnov test.
    """

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def detect(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> DriftResult:

        numerical_columns = reference_data.select_dtypes(
            include="number"
        ).columns

        if len(numerical_columns) == 0:
            return DriftResult(
                drift_detected=False,
                drift_score=0.0,
                method="ks",
            )

        p_values = []

        for column in numerical_columns:

            if column not in current_data.columns:
                continue

            reference_values = reference_data[column].dropna()
            current_values = current_data[column].dropna()

            if len(reference_values) == 0 or len(current_values) == 0:
                continue

            _, p_value = ks_2samp(
                reference_values,
                current_values,
            )

            p_values.append(p_value)

        if not p_values:
            return DriftResult(
                drift_detected=False,
                drift_score=0.0,
                method="ks",
            )

        minimum_p_value = min(p_values)

        return DriftResult(
            drift_detected=bool(minimum_p_value < self.threshold),
            drift_score=float(1.0 - minimum_p_value),
            method="ks",
        )