from pathlib import Path

import pandas as pd


def load_movielens_ratings(path: str | Path) -> pd.DataFrame:
    """
    Load MovieLens 1M ratings and convert them into
    the interaction format used by the recommender.

    MovieLens format:
        UserID::MovieID::Rating::Timestamp

    Our format:
        user_id
        item_id
        reward
        timestamp

    Ratings >= 4 are treated as positive interactions.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"MovieLens ratings file not found: {path}"
        )

    data = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=[
            "user_id",
            "item_id",
            "rating",
            "timestamp",
    ])

    # Convert the timestamp from Unix seconds
    # into a pandas datetime.
    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        unit="s",
        utc=True,
    )

    # Convert ratings into a binary interaction signal.
    data["reward"] = (
        data["rating"] >= 4
    ).astype(int)

    return data[
        [
            "user_id",
            "item_id",
            "rating",
            "reward",
            "timestamp",
        ]
    ]