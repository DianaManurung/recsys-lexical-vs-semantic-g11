"""Run the checks in src/leakage_check.py on the four models' validation top-10.

Writes ``outputs/logs/leakage_check_val.json``.

Usage (from the repository root):
    python -m src.leakage_val
"""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.leakage_check import (  # noqa: E402
    LeakageError, check_features, check_recommendations, check_same_candidates,
    run_static_checks,
)
from src.shared_eval import build_protocol, check_top_k, evaluable_users_rule  # noqa: E402
from src.user_groups import build_models  # noqa: E402

OUTPUT_PATH = ROOT / "outputs" / "logs" / "leakage_check_val.json"
K = 10
# Columns read by the models: interaction columns and the item text column.
ITEM_COLUMNS = ["item_id", "content_text_basic"]


def run(models=None, data=None, recs=None, save=True):
    train, val, cand_set, seen, relevant = data or build_protocol(str(ROOT / "data") + "/")
    if models is None:
        models, _ = build_models(train, cand_set)
    if recs is None:
        recs = {name: {u: m.recommend(u, seen, K) for u in relevant}
                for name, m in models.items()}

    universes = {"Popularity (fit input)": set(cand_set)}
    universes.update({name: set(models[name].candidate_items)
                      for name in ("MF", "TF-IDF", "Embedding") if name in models})

    per_model = {}
    for name, model_recs in recs.items():
        try:
            ok, message = check_recommendations(model_recs, cand_set, seen), ""
        except LeakageError as e:
            ok, message = False, str(e)
        per_model[name] = {"users_checked": len(model_recs),
                           "check_recommendations": ok, "message": message,
                           **check_top_k(model_recs, cand_set, seen, K)}

    result = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "evaluable_users_rule": evaluable_users_rule(),
        "n_evaluable_users": len(relevant),
        "static_checks": run_static_checks(train, val, relevant, cand_set, root=str(ROOT)),
        "check_features": check_features(list(train.columns) + ITEM_COLUMNS),
        "feature_columns": list(train.columns) + ITEM_COLUMNS,
        "check_same_candidates": check_same_candidates(universes),
        "candidate_universe_sizes": {k: len(v) for k, v in universes.items()},
        "per_model": per_model,
        "total_violations": sum(
            r["wrong_length"] + r["duplicate_items"] + r["seen_items"]
            + r["non_candidate_items"] + (not r["check_recommendations"])
            for r in per_model.values()),
    }
    if save:
        OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
