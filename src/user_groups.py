"""User activity groups for RQ2 (separate analysis track).

Each user is labelled by the number of interactions in the TRAIN split only,
using the bounds in ``configs/user_groups.json``. The groups are used to break
down metrics after evaluation; they never filter the main evaluation.
Models: Popularity, MF, TF-IDF CBF and Embedding CBF, each through its own
``recommend`` method with unchanged parameters.

Usage (from the repository root):
    python -m src.user_groups
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared_eval import (  # noqa: E402
    build_protocol, evaluable_users_rule, evaluate_per_user, summarize,
)

CONFIG_PATH = ROOT / "configs" / "user_groups.json"
OUTPUT_PATH = ROOT / "outputs" / "metrics" / "user_groups_val.json"
K = 10


def load_group_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def train_activity(train):
    """Number of train interactions per user (all ratings)."""
    return train.groupby("user_id").size().rename("train_interactions")


def assign_groups(activity, config):
    """Map each user's train interaction count to a group label."""
    labels = pd.Series(pd.NA, index=activity.index, dtype="object")
    for group in config["groups"]:
        low = -np.inf if group["min"] is None else group["min"]
        high = np.inf if group["max"] is None else group["max"]
        labels[(activity >= low) & (activity <= high)] = group["label"]
    if labels.isna().any():
        raise ValueError("Some users do not fall into any group")
    order = [g["label"] for g in config["groups"]]
    return pd.Categorical(labels, categories=order, ordered=True)


def group_table(per_user_by_model, groups, k=K):
    """Per-group metrics for each model.

    ``per_user_by_model``: ``{model_name: evaluate_per_user DataFrame}``.
    ``groups``: Series ``user_id -> label`` covering all evaluated users.
    """
    rows = []
    for model_name, per_user in per_user_by_model.items():
        labelled = per_user.assign(group=per_user["user_id"].map(groups))
        for label in list(groups.cat.categories) + ["all"]:
            part = labelled if label == "all" else labelled[labelled["group"] == label]
            if part.empty:
                continue
            rows.append({"group": label, "model": model_name, **summarize(part, k)})
    return pd.DataFrame(rows)


# Parameters as in notebooks/validation_eval.ipynb and the stored configs.
MF_PARAMS = dict(n_factors=32, n_epochs=10, learning_rate=0.01,
                 regularization=0.02, min_rating=4, random_state=42,
                 item_bias_weight=0.0)
TFIDF_PARAMS = dict(stop_words="english", min_df=2, max_df=0.8,
                    sublinear_tf=True, min_rating=4)
EMB_PARAMS = dict(min_rating=4)
MODEL_NAMES = ["Popularity", "MF", "TF-IDF", "Embedding"]


class RecordingModel:
    """Passes ``recommend`` through unchanged and keeps each top-k list."""

    def __init__(self, model):
        self.model = model
        self.recs = {}

    def recommend(self, user_id, seen, k=10):
        out = self.model.recommend(user_id, seen, k)
        self.recs[user_id] = list(out)
        return out


def build_models(train, cand_set, names=MODEL_NAMES):
    """Fit the requested models with their existing classes and parameters."""
    from src.content_based import TfidfContentRecommender
    from src.embedding_cache import ensure_item_embeddings
    from src.matrix_factorization import MatrixFactorizationRecommender
    from src.popularity import PopularityRecommender
    from src.semantic_based import SemanticEmbeddingRecommender

    builders = {
        "Popularity": lambda: PopularityRecommender(min_rating=4).fit(train, cand_set),
        "MF": lambda: MatrixFactorizationRecommender(**MF_PARAMS).fit(train, cand_set),
        "TF-IDF": lambda: TfidfContentRecommender(**TFIDF_PARAMS).fit(train, cand_set),
        "Embedding": lambda: SemanticEmbeddingRecommender(**EMB_PARAMS).fit(train, cand_set),
    }
    cache_info = ensure_item_embeddings() if "Embedding" in names else None
    return {name: builders[name]() for name in names}, cache_info


def run(models=None, data=None, save=True):
    """Evaluate each model per user and break the metrics down by group.

    ``data`` may be the tuple returned by ``build_protocol`` and ``models`` a
    ``{name: fitted model}`` dict, to reuse objects from a notebook. Returns
    ``(result, table, per_user, groups, recs)``; ``recs`` holds every top-10
    list for the leakage checks.
    """
    rule = evaluable_users_rule()
    train, _, cand_set, seen, relevant = data or build_protocol(str(ROOT / "data") + "/")
    cache_info = None
    if models is None:
        models, cache_info = build_models(train, cand_set)

    config = load_group_config()
    activity = train_activity(train)
    groups = pd.Series(assign_groups(activity, config), index=activity.index)

    per_user, recs = {}, {}
    for name, model in models.items():
        recorder = RecordingModel(model)
        per_user[name] = evaluate_per_user(recorder, relevant, seen, K)
        recs[name] = recorder.recs
    table = group_table(per_user, groups)

    eval_users = pd.Index(list(relevant))
    positive_users = set(train.loc[train["rating"] >= 4, "user_id"])
    counts = (
        pd.DataFrame({"group": groups.reindex(eval_users).to_numpy(),
                      "no_positive_train_item": [u not in positive_users for u in eval_users]})
        .groupby("group", observed=False)
        .agg(n_evaluable_users=("group", "size"),
             n_without_positive_train_item=("no_positive_train_item", "sum"))
    )

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "group_bounds": config["groups"],
        "evaluation": "src.evaluate.evaluate_per_user; group means summed and divided "
                      "by n as in src.evaluate.evaluate_model",
        "evaluable_users_rule": rule,
        "n_evaluable_users": int(len(relevant)),
        "models": {
            "Popularity": {"min_rating": 4},
            "MF": MF_PARAMS,
            "TF-IDF": TFIDF_PARAMS,
            "Embedding": {**EMB_PARAMS, "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"},
        },
        "unique_items_in_top10": {
            name: len({i for r in model_recs.values() for i in r})
            for name, model_recs in recs.items()
        },
        "n_train_users_per_group": {
            str(k): int(v) for k, v in groups.value_counts(sort=False).items()
        },
        "evaluable_users_per_group": {
            str(g): {"n_evaluable_users": int(r.n_evaluable_users),
                     "n_without_positive_train_item": int(r.n_without_positive_train_item)}
            for g, r in counts.iterrows()
        },
        "metrics": table.to_dict(orient="records"),
    }
    if cache_info is not None:
        result["embedding_cache"] = cache_info
    if save:
        OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, table, per_user, groups, recs


if __name__ == "__main__":
    _, table, _, _, _ = run()
    print(table.to_string(index=False))
