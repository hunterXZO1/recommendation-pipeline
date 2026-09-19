from pathlib import Path
import pandas as pd


def load_interactions(path: str | Path) -> pd.DataFrame:
    """
    Load user interaction data from a CSV file.

    Parameters
    ----------
    path:
        Path to the interaction CSV.

    Returns
    -------
    pd.DataFrame
        Loaded interaction data.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Interaction data not found: {path}")

    return pd.read_csv(path)