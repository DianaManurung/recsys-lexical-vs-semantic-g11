"""User activity groups for RQ2 (separate analysis track).

Each user is labelled by the number of interactions in the TRAIN split only,
using the bounds in ``configs/user_groups.json``. The groups are used to break
down metrics after evaluation; they never filter the main evaluation.

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

from src.shared_eval import build_protocol, evaluate_per_user, summarize  # noqa: E402

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


def build_models(train, cand_set):
    """TF-IDF baseline (default parameters) and the existing embedding model."""
    from src.content_based import TfidfContentRecommender, load_item_text
    from src.embedding_cache import ensure_item_embeddings
    from src.semantic_based import SemanticEmbeddingRecommender

    item_text = load_item_text(ROOT / "data" / "items.csv", cand_set)
    tfidf = TfidfContentRecommender(
        stop_words="english", min_df=2, max_df=0.8,
        sublinear_tf=True, min_rating=4,
    ).fit(train, cand_set, item_text)
    cache_info = ensure_item_embeddings()
    embedding = SemanticEmbeddingRecommender(min_rating=4).fit(train, cand_set)
    return {"TF-IDF": tfidf, "Embedding": embedding}, cache_info


def run(models=None, data=None, save=True):
    """Evaluate both models per user and break the metrics down by group.

    ``data`` may be the tuple returned by ``build_protocol`` and ``models`` a
    ``{name: fitted model}`` dict, to reuse objects from a notebook.
    """
    train, _, cand_set, seen, relevant = data or build_protocol(str(ROOT / "data") + "/")
    cache_info = None
    if models is None:
        models, cache_info = build_models(train, cand_set)

    config = load_group_config()
    activity = train_activity(train)
    groups = pd.Series(assign_groups(activity, config), index=activity.index)

    per_user = {name: evaluate_per_user(m, relevant, seen, K) for name, m in models.items()}
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
        "evaluation": "src/shared_eval.py (same protocol as the main evaluation)",
        "n_evaluable_users": int(len(relevant)),
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
    return result, table, per_user, groups


if __name__ == "__main__":
    _, table, _, _ = run()
    print(table.to_string(index=False))
