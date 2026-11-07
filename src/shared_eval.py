"""Shared evaluation protocol used by the CP3 experiments.

Temporary wrapper until Diana's protocol and evaluation module is merged
(branch feature/cp3-protocol-evaluate). The function names, arguments and
return values follow ``src.protocol.build_protocol`` and
``src.evaluate.evaluate_model`` / ``evaluate_per_user`` on that branch. The
computation is the one used by the TF-IDF notebook (``evaluate_at_k`` and
``hit_rate_at_k``), so the stored TF-IDF metrics are reproduced exactly.

After the merge, replace the body of this module with:

    from src.protocol import build_protocol
    from src.evaluate import evaluate_model, evaluate_per_user
"""

import numpy as np
import pandas as pd

from src.data_loader import load_data, relevant_items_val

MIN_RATING = 4
TOP_K = 10


def build_protocol(data_dir="data/", min_rating=MIN_RATING):
    """Return ``(train, val, cand_set, seen, relevant)``.

    ``relevant`` maps each evaluable user to the set of validation items rated
    ``>= min_rating`` that are warm candidates; its keys are the evaluable
    users. Only train and validation are read.
    """
    train, val, cand_set, seen = load_data(data_dir)
    relevant = relevant_items_val(val, cand_set, seen, min_rating)
    return train, val, cand_set, seen, relevant


def _user_metrics(recs, relevant_items, k):
    """Recall, NDCG and hit for one user, as in the TF-IDF notebook."""
    hits = [item in relevant_items for item in recs[:k]]
    recall = sum(hits) / len(relevant_items)
    dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1 / np.log2(rank + 2) for rank in range(ideal_hits))
    ndcg = dcg / idcg if idcg else 0.0
    return recall, ndcg, float(any(hits))


def evaluate_per_user(model, relevant, seen, k=TOP_K):
    """Per-user metrics as a DataFrame with columns
    ``user_id, n_relevant, recall, ndcg, hit``."""
    rows = []
    for user_id, relevant_items in relevant.items():
        recs = model.recommend(user_id, seen, k=k)
        recall, ndcg, hit = _user_metrics(recs, relevant_items, k)
        rows.append({
            "user_id": user_id,
            "n_relevant": len(relevant_items),
            "recall": recall,
            "ndcg": ndcg,
            "hit": hit,
        })
    return pd.DataFrame(rows)


def summarize(per_user, k=TOP_K):
    """Average a per-user DataFrame into the ``evaluate_model`` format."""
    if per_user.empty:
        raise ValueError("No evaluable users.")
    return {
        f"Recall@{k}": float(np.mean(per_user["recall"].to_numpy())),
        f"NDCG@{k}": float(np.mean(per_user["ndcg"].to_numpy())),
        f"HitRate@{k}": float(np.mean(per_user["hit"].to_numpy())),
        "n_users": int(len(per_user)),
    }


def evaluate_model(model, relevant, seen, k=TOP_K):
    """Mean Recall@k, NDCG@k and HitRate@k over all evaluable users.

    ``model`` needs a ``recommend(user_id, seen, k)`` method. Returns
    ``{"Recall@k", "NDCG@k", "HitRate@k", "n_users"}``.
    """
    return summarize(evaluate_per_user(model, relevant, seen, k), k)


def check_top_k(recs_by_user, cand_set, seen, k=TOP_K):
    """Count rule violations: exactly ``k`` unique items, none seen in train,
    all from the candidate set."""
    counts = {"wrong_length": 0, "duplicate_items": 0,
              "seen_items": 0, "non_candidate_items": 0}
    for user_id, recs in recs_by_user.items():
        if len(recs) != k:
            counts["wrong_length"] += 1
        if len(set(recs)) != len(recs):
            counts["duplicate_items"] += 1
        if set(recs) & seen.get(user_id, set()):
            counts["seen_items"] += 1
        if not set(recs) <= cand_set:
            counts["non_candidate_items"] += 1
    return counts
