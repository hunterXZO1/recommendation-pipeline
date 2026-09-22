from pathlib import Path

from src.data.movielens import load_movielens_ratings


def test_load_movielens():

    path = Path("data/ml-1m/ratings.dat")

    data = load_movielens_ratings(path)

    assert not data.empty

    assert "user_id" in data.columns
    assert "item_id" in data.columns
    assert "rating" in data.columns
    assert "reward" in data.columns
    assert "timestamp" in data.columns


def test_reward_conversion():

    path = Path("data/ml-1m/ratings.dat")

    data = load_movielens_ratings(path)

    assert set(data["reward"].unique()).issubset({0, 1})