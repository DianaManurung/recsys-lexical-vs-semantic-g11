
"""Semantic embedding-based recommender using precomputed item vectors."""

from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_EMBEDDINGS_PATH = (
    ROOT_DIR / "outputs" / "embeddings" / "item_embeddings.npy"
)
DEFAULT_ITEM_IDS_PATH = (
    ROOT_DIR / "outputs" / "embeddings" / "item_ids.npy"
)


class SemanticEmbeddingRecommender:
    """Rank candidate items by cosine similarity to user profiles."""

    def __init__(self, min_rating=4):
        self.min_rating = min_rating

        self.item_ids = None
        self.item_matrix = None
        self.item_index = {}

        self.candidate_items = []
        self.cand_matrix = None
        self.cand_pos = {}

        self.user_items = {}
        self.ranked_items = []

    def fit(
        self,
        train,
        cand_set,
        embeddings_path=DEFAULT_EMBEDDINGS_PATH,
        item_ids_path=DEFAULT_ITEM_IDS_PATH,
    ):
        """Load item embeddings and build positive user-item profiles."""

        if train.empty:
            raise ValueError("train must contain at least one interaction")

        cand_set = set(cand_set)

        if not cand_set:
            raise ValueError("cand_set must not be empty")

        # Load the precomputed embeddings and their corresponding IDs.
        self.item_ids = np.load(
            item_ids_path,
            allow_pickle=False,
        )

        self.item_matrix = np.load(
            embeddings_path,
            allow_pickle=False,
        )

        if self.item_matrix.ndim != 2:
            raise ValueError("Embedding matrix must be two-dimensional")

        if len(self.item_ids) != len(self.item_matrix):
            raise ValueError("Item IDs and embeddings have different lengths")

        if len(set(self.item_ids.tolist())) != len(self.item_ids):
            raise ValueError("Duplicate item IDs found in embedding files")

        if not np.isfinite(self.item_matrix).all():
            raise ValueError("Embedding matrix contains invalid values")

        if not np.allclose(
            np.linalg.norm(self.item_matrix, axis=1),
            1.0,
            atol=1e-5,
        ):
            raise ValueError("Item embeddings are not L2-normalized")

        self.item_index = {
            item_id: idx
            for idx, item_id in enumerate(self.item_ids.tolist())
        }

        # Require exactly the same candidate set as the embedding files.
        embedding_ids = set(self.item_index)

        if cand_set != embedding_ids:
            missing = len(cand_set - embedding_ids)
            extra = len(embedding_ids - cand_set)

            raise ValueError(
                "Candidate and embedding IDs do not match. "
                f"Missing embeddings: {missing}; extra embeddings: {extra}"
            )

        # Store positive train items for each user.
        pos = train[train["rating"] >= self.min_rating]
        pos = pos[pos["item_id"].isin(self.item_index)]

        self.user_items = (
            pos.groupby("user_id")["item_id"]
            .apply(
                lambda s: np.fromiter(
                    (self.item_index[item] for item in s),
                    dtype=np.int32,
                )
            )
            .to_dict()
        )

        # Popularity fallback, following the TF-IDF baseline.
        positive_counts = train.loc[
            train["rating"] >= self.min_rating,
            "item_id",
        ].value_counts()

        self.ranked_items = [
            item for item in positive_counts.index
            if item in cand_set
        ]

        # Keep candidate ordering consistent with the item IDs.
        self.candidate_items = sorted(cand_set)

        rows = np.array(
            [
                self.item_index[item]
                for item in self.candidate_items
            ],
            dtype=np.int32,
        )

        self.cand_matrix = self.item_matrix[rows]

        self.cand_pos = {
            item: idx
            for idx, item in enumerate(self.candidate_items)
        }

        return self

    def user_profile(self, user_id):
        """Return the normalized mean embedding of positive train items."""

        idx = self.user_items.get(user_id)

        if idx is None or len(idx) == 0:
            return None

        profile = self.item_matrix[idx].mean(axis=0)

        norm = np.linalg.norm(profile)

        if norm <= 1e-12:
            return None

        return profile / norm

    def recommend(self, user_id, seen, k=10):
        """Return the top-k most similar unseen candidate items."""

        if k <= 0:
            return []

        seen_items = seen.get(user_id, set())

        profile = self.user_profile(user_id)

        # No usable profile: fall back to train popularity.
        if profile is None:
            return [
                item for item in self.ranked_items
                if item not in seen_items
            ][:k]

        # Unit-normalized item vectors and profile:
        # dot product equals cosine similarity.
        scores = self.cand_matrix @ profile

        for item in seen_items:
            pos = self.cand_pos.get(item)

            if pos is not None:
                scores[pos] = -np.inf

        # Select positive similarity scores first.
        top = min(k, int((scores > 0).sum()))

        if top:
            part = np.argpartition(-scores, top - 1)[:top]

            order = part[
                np.argsort(-scores[part], kind="stable")
            ]

            out = [
                self.candidate_items[i]
                for i in order
            ]
        else:
            out = []

        # Fill insufficient results using popularity.
        if len(out) < k:
            chosen = set(out)

            for item in self.ranked_items:
                if item not in seen_items and item not in chosen:
                    out.append(item)
                    chosen.add(item)

                    if len(out) == k:
                        break

        return out
