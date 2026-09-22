import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerRecommender(nn.Module):
    """
    Two-tower recommendation model.

    One tower learns a representation for users.
    The other tower learns a representation for items.

    The dot product between the two embeddings is the
    recommendation score.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim
        )

        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim
        )

    def encode_users(self, user_ids):
        """
        Convert user IDs into normalized user embeddings.
        """
        embeddings = self.user_embedding(user_ids)
        return F.normalize(embeddings, p=2, dim=1)

    def encode_items(self, item_ids):
        """
        Convert item IDs into normalized item embeddings.
        """
        embeddings = self.item_embedding(item_ids)
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, user_ids, item_ids):
        """
        Calculate the recommendation score for
        user-item pairs.
        """

        user_vectors = self.encode_users(user_ids)
        item_vectors = self.encode_items(item_ids)

        scores = (user_vectors * item_vectors).sum(dim=1)

        return scores

    def recommend_scores(self, user_id, item_ids):
        """
        Calculate scores between one user and multiple items.
        """

        user_tensor = torch.tensor(
            [user_id],
            dtype=torch.long
        )

        item_tensor = torch.tensor(
            item_ids,
            dtype=torch.long
        )

        user_vector = self.encode_users(
            user_tensor
        )

        item_vectors = self.encode_items(
            item_tensor
        )

        scores = user_vector @ item_vectors.T

        return scores.squeeze(0)