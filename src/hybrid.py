"""Content + popularity hybrid for E5 (TF-IDF) and E6 (semantic embedding).

Score rule from docs/experiment_matrix.md section 4 and
configs/hybrid_search.json / configs/hybrid_embedding_search.json:

    s(u, i) = alpha * s_content(u, i) + (1 - alpha) * s_pop(i)

- ``s_content``: cosine similarity between the user profile and the item
  vector of a fitted ``TfidfContentRecommender`` or
  ``SemanticEmbeddingRecommender``.
- ``s_pop``: number of positive (rating >= min_rating) train interactions.
- Both scores are min-max normalized separately for each user over the
  unseen warm candidates.
- A user without a usable content profile gets content score 0, so the
  popularity score alone decides the ranking.
"""

import numpy as np


def _minmax(values):
    lo, hi = values.min(), values.max()
    if hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


class ContentPopularityHybrid:
    """Wrap a fitted content model and blend its scores with popularity."""

    def __init__(self, content_model, alpha=0.5, min_rating=4):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.content = content_model
        self.alpha = alpha
        self.min_rating = min_rating
        self.candidate_items = []
        self.cand_pos = {}
        self.pop_scores = None

    def fit(self, train):
        """Popularity counts over the content model's candidate order."""
        self.candidate_items = list(self.content.candidate_items)
        self.cand_pos = dict(self.content.cand_pos)
        counts = train.loc[train["rating"] >= self.min_rating, "item_id"].value_counts()
        self.pop_scores = np.array(
            [float(counts.get(item, 0)) for item in self.candidate_items]
        )
        return self

    def scores(self, user_id, seen):
        """Hybrid scores over all candidates; seen items are ``-inf``."""
        unseen = np.ones(len(self.candidate_items), dtype=bool)
        for item in seen.get(user_id, set()):
            pos = self.cand_pos.get(item)
            if pos is not None:
                unseen[pos] = False

        profile = self.content.user_profile(user_id)
        content = np.zeros(len(self.candidate_items))
        if profile is not None:
            raw = np.asarray(self.content.cand_matrix @ profile, dtype=np.float64).ravel()
            content[unseen] = _minmax(raw[unseen])
        pop = np.zeros(len(self.candidate_items))
        pop[unseen] = _minmax(self.pop_scores[unseen])

        out = self.alpha * content + (1.0 - self.alpha) * pop
        out[~unseen] = -np.inf
        return out

    def recommend(self, user_id, seen, k=10):
        """Top-k unseen candidates; exact ties keep candidate (item_id) order."""
        if k <= 0:
            return []
        s = self.scores(user_id, seen)
        n_valid = int(np.isfinite(s).sum())
        k = min(k, n_valid)
        if k == 0:
            return []
        # Stable full sort: exact ties keep candidate index order everywhere,
        # including at the top-k boundary.
        order = np.argsort(-s, kind="stable")[:k]
        return [self.candidate_items[i] for i in order]
