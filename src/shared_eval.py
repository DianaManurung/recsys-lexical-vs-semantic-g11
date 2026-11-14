"""Evaluation entry point for Tiffani's scripts and notebooks.

Metrics come from Diana's ``src/evaluate.py``. The evaluable-user list is
chosen with one setting, ``evaluable_users`` in ``configs/evaluation.json``:

- ``"official"``: ``src.protocol.build_protocol`` (11,659 users, identical to
  ``data/evaluable_users_validation_warm.csv``).
- ``"group_rule"``: ``load_data`` + ``relevant_items_val`` without the positive
  train item filter (11,850 users, the rule used before W10).

Only train and validation are read.
"""

import json
from pathlib import Path

from src import protocol
from src.data_loader import load_data, relevant_items_val
from src.evaluate import evaluate_model, evaluate_per_user  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "evaluation.json"
RULES = ("official", "group_rule")
MIN_RATING = protocol.MIN_RATING
TOP_K = protocol.TOP_K


def evaluable_users_rule(path=CONFIG_PATH):
    """Rule name set in ``configs/evaluation.json``."""
    rule = json.loads(Path(path).read_text(encoding="utf-8"))["evaluable_users"]
    if rule not in RULES:
        raise ValueError(f"evaluable_users must be one of {RULES}, got {rule!r}")
    return rule


def build_protocol(data_dir="data/", min_rating=MIN_RATING, rule=None):
    """Return ``(train, val, cand_set, seen, relevant)`` for the chosen rule.

    ``rule`` overrides the config file when given.
    """
    rule = rule or evaluable_users_rule()
    if rule == "official":
        return protocol.build_protocol(data_dir, min_rating)
    train, val, cand_set, seen = load_data(data_dir)
    relevant = relevant_items_val(val, cand_set, seen, min_rating)
    return train, val, cand_set, seen, relevant


def summarize(per_user, k=TOP_K):
    """Average an ``evaluate_per_user`` DataFrame.

    Values are summed in user order and divided by n, as in
    ``src.evaluate.evaluate_model``, so the overall row matches it exactly.
    """
    n = len(per_user)
    if n == 0:
        raise ValueError("No evaluable users.")
    return {
        f"Recall@{k}": sum(per_user["recall"].tolist()) / n,
        f"NDCG@{k}": sum(per_user["ndcg"].tolist()) / n,
        f"HitRate@{k}": sum(per_user["hit"].tolist()) / n,
        "n_users": int(n),
    }


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
