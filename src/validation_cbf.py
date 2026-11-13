"""Validation metrics for TF-IDF CBF and Embedding CBF (W10).

Both models are evaluated with ``src.evaluate.evaluate_model`` on the
evaluable users chosen in ``configs/evaluation.json``. Results overwrite
``outputs/metrics/tfidf_cbf_val.json`` and ``semantic_cbf_val.json``; the
values stored before the W10 re-evaluation are kept under ``"before_w10"``
(summary) and ``"before_w10_file"`` (full content).

Usage (from the repository root):
    python -m src.validation_cbf
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared_eval import build_protocol, evaluable_users_rule, evaluate_model  # noqa: E402

METRICS = ["Recall@10", "NDCG@10", "HitRate@10"]
OUTPUTS = {
    "TF-IDF": ROOT / "outputs" / "metrics" / "tfidf_cbf_val.json",
    "Embedding": ROOT / "outputs" / "metrics" / "semantic_cbf_val.json",
}
# Same parameters as outputs/configs/tfidf_cbf.json and semantic_cbf.json.
TFIDF_PARAMS = dict(stop_words="english", min_df=2, max_df=0.8,
                    sublinear_tf=True, min_rating=4)
EMB_PARAMS = dict(min_rating=4)


def build_models(train, cand_set):
    from src.content_based import TfidfContentRecommender
    from src.embedding_cache import ensure_item_embeddings
    from src.semantic_based import SemanticEmbeddingRecommender

    ensure_item_embeddings()
    return {
        "TF-IDF": TfidfContentRecommender(**TFIDF_PARAMS).fit(train, cand_set),
        "Embedding": SemanticEmbeddingRecommender(**EMB_PARAMS).fit(train, cand_set),
    }


def previous_content(path):
    """Content stored before W10; on a re-run, the kept ``before_w10_file``."""
    if not path.exists():
        return None
    old = json.loads(path.read_text(encoding="utf-8"))
    return old.get("before_w10_file", old) if "before_w10" in old else old


def run(models=None, data=None, save=True):
    rule = evaluable_users_rule()
    train, _, cand_set, seen, relevant = data or build_protocol(str(ROOT / "data") + "/")
    models = models or build_models(train, cand_set)
    positive_users = set(train.loc[train["rating"] >= 4, "user_id"])
    n_with_profile = sum(u in positive_users for u in relevant)

    results = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        m = evaluate_model(model, relevant, seen, k=10)
        prev = previous_content(OUTPUTS[name])
        out = {
            **{k: m[k] for k in METRICS},
            "n_eval_users": m["n_users"],
            "n_eval_users_with_profile": n_with_profile,
            "evaluable_users_rule": rule,
            "evaluator": "src.evaluate.evaluate_model",
            "protocol_builder": "src.shared_eval.build_protocol -> "
                                + ("src.protocol.build_protocol" if rule == "official"
                                   else "src.data_loader.relevant_items_val"),
            "params": TFIDF_PARAMS if name == "TF-IDF" else
                      {**EMB_PARAMS, "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"},
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        if prev is not None:
            out["before_w10"] = {
                **{k: prev.get(k) for k in METRICS},
                "n_eval_users": prev.get("n_eval_users"),
                "abs_diff": {k: abs(out[k] - prev[k]) for k in METRICS if k in prev},
            }
            out["before_w10_file"] = prev
        results[name] = out
        if save:
            OUTPUTS[name].write_text(json.dumps(out, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    for name, r in run().items():
        prev = r.get("before_w10", {})
        print(f"{name}: n={r['n_eval_users']}  " + "  ".join(
            f"{k}={r[k]:.6f} (prev {prev.get(k, float('nan')):.6f})" for k in METRICS))
