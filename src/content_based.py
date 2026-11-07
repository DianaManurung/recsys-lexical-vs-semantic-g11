"""TF-IDF content-based filtering over item text (``data/items.csv``)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_ITEMS_PATH = Path(__file__).resolve().parents[1] / "data" / "items.csv"


def load_item_text(path, cand_set, text_col="content_text_basic"):
    """Return ``{item_id: text}`` for warm candidates from ``items.csv``."""
    items = pd.read_csv(path, dtype={"item_id": str}, usecols=["item_id", text_col])
    items = items[items["item_id"].isin(cand_set)]
    return dict(zip(items["item_id"], items[text_col].fillna("")))


class TfidfContentRecommender:
    """Rank items by cosine similarity between TF-IDF item vectors and a user
    profile built from the items the user rated ``>= min_rating`` in train.

    Users without a usable profile fall back to train popularity, and
    candidates with a zero score are filled in popularity order.
    """

    def __init__(
        self,
        stop_words="english",
        min_df=2,
        max_df=0.8,
        sublinear_tf=True,
        min_rating=4,
        max_features=None,
        ngram_range=(1, 1),
    ):
        # max_features and ngram_range default to the TfidfVectorizer
        # defaults, so the baseline run is unchanged.
        self.stop_words = stop_words
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.min_rating = min_rating
        self.max_features = max_features
        self.ngram_range = tuple(ngram_range)
        self.vectorizer = None
        self.item_index = {}
        self.item_matrix = None
        self.cand_matrix = None
        self.cand_pos = {}
        self.user_items = {}
        self.ranked_items = []
        self.candidate_items = []

    def fit(self, train, cand_set, item_text=None):
        """Fit TF-IDF on ``item_text`` (``{item_id: text}``, read from
        ``data/items.csv`` when omitted) and store each user's positively
        rated train items."""
        if train.empty:
            raise ValueError("train must contain at least one interaction")
        if item_text is None:
            item_text = load_item_text(DEFAULT_ITEMS_PATH, cand_set)
        if not item_text:
            raise ValueError("item_text must contain at least one item")

        items = sorted(item_text)
        self.vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )
        # Rows are L2-normalized, so dot products are cosine similarities.
        matrix = self.vectorizer.fit_transform(item_text[i] for i in items).tocsr()
        self.item_index = {item: idx for idx, item in enumerate(items)}

        pos = train[train["rating"] >= self.min_rating]
        pos = pos[pos["item_id"].isin(self.item_index)]
        self.user_items = (
            pos.groupby("user_id")["item_id"]
            .apply(lambda s: np.fromiter((self.item_index[i] for i in s), dtype=np.int32))
            .to_dict()
        )

        positive_counts = train.loc[
            train["rating"] >= self.min_rating, "item_id"
        ].value_counts()
        self.ranked_items = [
            item for item in positive_counts.index if item in cand_set
        ]

        # Candidate rows; candidates without text get an all-zero row.
        self.candidate_items = sorted(cand_set)
        rows = np.array(
            [self.item_index.get(item, -1) for item in self.candidate_items]
        )
        has_row = rows >= 0
        self.item_matrix = matrix
        self.cand_matrix = (
            matrix[np.where(has_row, rows, 0)].multiply(has_row[:, None]).tocsr()
        )
        self.cand_pos = {item: idx for idx, item in enumerate(self.candidate_items)}
        return self

    def user_profile(self, user_id):
        """Mean TF-IDF vector of the user's positive train items, or ``None``."""
        idx = self.user_items.get(user_id)
        if idx is None or len(idx) == 0:
            return None
        return np.asarray(self.item_matrix[idx].mean(axis=0)).ravel()

    def recommend(self, user_id, seen, k=10):
        """Return up to ``k`` most similar unseen candidate items."""
        if k <= 0:
            return []
        seen_items = seen.get(user_id, set())
        profile = self.user_profile(user_id)
        if profile is None:
            return [item for item in self.ranked_items if item not in seen_items][:k]

        scores = self.cand_matrix @ profile
        for item in seen_items:
            pos = self.cand_pos.get(item)
            if pos is not None:
                scores[pos] = -np.inf

        top = min(k, int((scores > 0).sum()))
        if top:
            part = np.argpartition(-scores, top - 1)[:top]
            order = part[np.argsort(-scores[part], kind="stable")]
            out = [self.candidate_items[i] for i in order]
        else:
            out = []
        if len(out) < k:
            chosen = set(out)
            for item in self.ranked_items:
                if item not in seen_items and item not in chosen:
                    out.append(item)
                    if len(out) == k:
                        break
        return out
