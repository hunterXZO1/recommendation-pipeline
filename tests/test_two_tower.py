import torch

from src.models.two_tower import TwoTowerRecommender


def test_model_forward():

    model = TwoTowerRecommender(
        num_users=10,
        num_items=20,
        embedding_dim=64,
    )

    user_ids = torch.tensor([0, 1, 2])
    item_ids = torch.tensor([3, 4, 5])

    scores = model(
        user_ids,
        item_ids
    )

    assert scores.shape == (3,)


def test_user_embedding():

    model = TwoTowerRecommender(
        num_users=10,
        num_items=20,
        embedding_dim=64,
    )

    user_ids = torch.tensor([0, 1])

    embeddings = model.encode_users(
        user_ids
    )

    assert embeddings.shape == (2, 64)


def test_item_embedding():

    model = TwoTowerRecommender(
        num_users=10,
        num_items=20,
        embedding_dim=64,
    )

    item_ids = torch.tensor([0, 1, 2])

    embeddings = model.encode_items(
        item_ids
    )

    assert embeddings.shape == (3, 64)