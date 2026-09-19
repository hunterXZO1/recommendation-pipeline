import pandas as pd

from src.drift.detector import KSNumericalDriftDetector


def test_no_drift():

    reference = pd.DataFrame({
        "age": [20, 21, 22, 23, 24, 25],
    })

    current = pd.DataFrame({
        "age": [20, 21, 22, 23, 24, 25],
    })

    detector = KSNumericalDriftDetector()

    result = detector.detect(
        reference_data=reference,
        current_data=current,
    )

    assert result.drift_detected is False
    assert result.method == "ks"


def test_detect_drift():

    reference = pd.DataFrame({
        "age": [
            20, 21, 22, 23, 24,
            25, 26, 27, 28, 29,
        ],
    })

    current = pd.DataFrame({
        "age": [
            80, 81, 82, 83, 84,
            85, 86, 87, 88, 89,
        ],
    })

    detector = KSNumericalDriftDetector()

    result = detector.detect(
        reference_data=reference,
        current_data=current,
    )

    assert result.drift_detected is True
    assert result.method == "ks"


def test_no_numerical_columns():

    reference = pd.DataFrame({
        "category": ["A", "B", "A"],
    })

    current = pd.DataFrame({
        "category": ["A", "B", "B"],
    })

    detector = KSNumericalDriftDetector()

    result = detector.detect(
        reference_data=reference,
        current_data=current,
    )

    assert result.drift_detected is False