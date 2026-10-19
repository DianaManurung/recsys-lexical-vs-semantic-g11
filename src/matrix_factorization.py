"""Biased matrix factorization for explicit user-item ratings."""

import numpy as np


class MatrixFactorizationRecommender:
    """Learn user/item factors with regularized stochastic gradient descent.

    Ratings are modeled as ``global_mean + user_bias + item_bias + p dot q``.
    The class uses NumPy only and recommends from a supplied warm-item set.
    """

    def __init__(
        self,
        n_factors=32,
        n_epochs=10,
        learning_rate=0.01,
        regularization=0.02,
        min_rating=4,
        random_state=42,
    ):
        if n_factors < 1 or n_epochs < 1:
            raise ValueError("n_factors and n_epochs must be positive")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.min_rating = min_rating
        self.random_state = random_state
        self.user_index = {}
        self.item_index = {}
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = 0.0
        self.ranked_items = []

    def fit(self, train, cand_set):
        """Fit on ``user_id``, ``item_id``, ``rating`` columns."""
        if train.empty:
            raise ValueError("train must contain at least one interaction")

        users = train["user_id"].drop_duplicates().tolist()
        items = train["item_id"].drop_duplicates().tolist()
        self.user_index = {user: idx for idx, user in enumerate(users)}
        self.item_index = {item: idx for idx, item in enumerate(items)}

        user_ids = train["user_id"].map(self.user_index).to_numpy(dtype=np.int32)
        item_ids = train["item_id"].map(self.item_index).to_numpy(dtype=np.int32)
        ratings = train["rating"].to_numpy(dtype=np.float64)
        self.global_mean = float(ratings.mean())

        rng = np.random.default_rng(self.random_state)
        scale = 0.1 / np.sqrt(self.n_factors)
        self.user_factors = rng.normal(
            0.0, scale, (len(users), self.n_factors)
        )
        self.item_factors = rng.normal(
            0.0, scale, (len(items), self.n_factors)
        )
        self.user_bias = np.zeros(len(users), dtype=np.float64)
        self.item_bias = np.zeros(len(items), dtype=np.float64)

        for _ in range(self.n_epochs):
            for row in rng.permutation(len(ratings)):
                u, i, actual = user_ids[row], item_ids[row], ratings[row]
                pu = self.user_factors[u].copy()
                qi = self.item_factors[i].copy()
                error = actual - (
                    self.global_mean + self.user_bias[u] + self.item_bias[i]
                    + np.dot(pu, qi)
                )
                self.user_bias[u] += self.learning_rate * (
                    error - self.regularization * self.user_bias[u]
                )
                self.item_bias[i] += self.learning_rate * (
                    error - self.regularization * self.item_bias[i]
                )
                self.user_factors[u] += self.learning_rate * (
                    error * qi - self.regularization * pu
                )
                self.item_factors[i] += self.learning_rate * (
                    error * pu - self.regularization * qi
                )

        # Keep a deterministic fallback ordering for users/items without factors.
        positive_counts = train.loc[
            train["rating"] >= self.min_rating, "item_id"
        ].value_counts()
        self.ranked_items = [
            item for item in positive_counts.index if item in cand_set
        ]
        self.candidate_items = sorted(
            item for item in cand_set if item in self.item_index
        )
        return self

    def recommend(self, user_id, seen, k=10):
        """Return up to ``k`` highest predicted unseen warm-item scores."""
        if k <= 0:
            return []
        seen_items = seen.get(user_id, set())
        if user_id not in self.user_index:
            return [item for item in self.ranked_items if item not in seen_items][:k]

        u = self.user_index[user_id]
        candidates = [
            item for item in self.candidate_items
            if item not in seen_items and item in self.item_index
        ]
        if not candidates:
            return []
        item_ids = np.fromiter(
            (self.item_index[item] for item in candidates), dtype=np.int32
        )
        scores = (
            self.global_mean + self.user_bias[u]
            + self.item_bias[item_ids]
            + self.item_factors[item_ids] @ self.user_factors[u]
        )
        order = np.argsort(-scores, kind="stable")[:k]
        return [candidates[idx] for idx in order]
