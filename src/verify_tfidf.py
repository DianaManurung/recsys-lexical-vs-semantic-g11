"""Re-run the TF-IDF baseline and compare with the stored metrics.

Usage (from the repository root):
    python -m src.verify_tfidf
"""

import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.content_based import TfidfContentRecommender, load_item_text  # noqa: E402
from src.shared_eval import build_protocol, evaluate_model  # noqa: E402

STORED = ROOT / "outputs" / "metrics" / "tfidf_cbf_val.json"
OUT = ROOT / "outputs" / "logs" / "tfidf_cbf_recheck.json"
METRICS = ["Recall@10", "NDCG@10", "HitRate@10"]


def main():
    train, _, cand_set, seen, relevant = build_protocol(str(ROOT / "data") + "/")
    item_text = load_item_text(ROOT / "data" / "items.csv", cand_set)

    t0 = time.perf_counter()
    # Same parameters as notebooks/content_based_tfidf_demo.ipynb.
    model = TfidfContentRecommender(
        stop_words="english", min_df=2, max_df=0.8,
        sublinear_tf=True, min_rating=4,
    ).fit(train, cand_set, item_text)
    got = evaluate_model(model, relevant, seen, k=10)
    seconds = time.perf_counter() - t0

    stored = json.loads(STORED.read_text(encoding="utf-8"))
    rows = {
        m: {"stored": stored[m], "rerun": got[m], "abs_diff": abs(stored[m] - got[m])}
        for m in METRICS
    }
    result = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "stored_file": STORED.relative_to(ROOT).as_posix(),
        "n_users": {"stored": stored["n_eval_users"], "rerun": got["n_users"]},
        "vocab_size": len(model.vectorizer.vocabulary_),
        "metrics": rows,
        "identical": all(r["abs_diff"] == 0 for r in rows.values())
        and stored["n_eval_users"] == got["n_users"],
        "seconds": round(seconds, 1),
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
        },
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
