"""Grid search runner for ``configs/*_search.json`` (validation only).

Every combination of the ``grid`` section is merged with ``fixed``, fitted on
train, evaluated on validation with ``src/shared_eval.py``, and appended as one
row to ``outputs/logs/experiment_log.csv``.

Usage (from the repository root):
    python -m src.run_grid configs/tfidf_search.json --list
    python -m src.run_grid configs/tfidf_search.json --only 0 1
    python -m src.run_grid configs/tfidf_search.json          # full grid
"""

import argparse
import csv
import itertools
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared_eval import build_protocol, evaluate_model  # noqa: E402

LOG_PATH = ROOT / "outputs" / "logs" / "experiment_log.csv"
LOG_COLUMNS = [
    "run_id", "timestamp", "model", "search_file", "combo_index", "params",
    "commit", "dirty", "selection_metric", "Recall@10", "NDCG@10",
    "HitRate@10", "n_users", "seconds",
]


def load_search(path):
    search = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("model", "fixed", "grid"):
        if key not in search:
            raise ValueError(f"{path}: missing '{key}'")
    return search


def expand_grid(grid):
    """All combinations of the grid values, in the order given in the file."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def git_state():
    def git(*args):
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip()
    return git("rev-parse", "--short", "HEAD"), bool(git("status", "--porcelain", "src", "configs"))


def build_tfidf(params, train, cand_set):
    from src.content_based import TfidfContentRecommender, load_item_text

    item_text = load_item_text(ROOT / "data" / "items.csv", cand_set,
                               text_col=params["text_field"])
    return TfidfContentRecommender(
        stop_words=params["stop_words"],
        min_df=params["min_df"],
        max_df=params["max_df"],
        sublinear_tf=params["sublinear_tf"],
        min_rating=params["min_rating"],
        max_features=params["max_features"],
        ngram_range=params["ngram_range"],
    ).fit(train, cand_set, item_text)


def build_embedding(params, train, cand_set):
    """Encode with src/semantic_embedding.py, one cache folder per model."""
    from src import semantic_embedding as se
    from src.semantic_based import SemanticEmbeddingRecommender

    if params["text_field"] != se.TEXT_COL:
        raise ValueError(f"text_field must be {se.TEXT_COL}")
    slug = params["model_name"].split("/")[-1]
    cache = ROOT / "outputs" / "embeddings" / "grid" / f"{slug}_len{params['max_seq_length']}"
    emb_path, ids_path = cache / "item_embeddings.npy", cache / "item_ids.npy"
    if not (emb_path.exists() and ids_path.exists()):
        item_text = se.load_item_text(se.DEFAULT_ITEMS_PATH, cand_set)
        item_ids, embeddings, stats = se.build_item_embeddings(
            item_text, model_name=params["model_name"],
            batch_size=params["batch_size"], device=params["device"],
        )
        # semantic_embedding.py uses the model's default max_seq_length.
        if stats["max_seq_length"] != params["max_seq_length"]:
            raise ValueError(
                f"{params['model_name']} uses max_seq_length "
                f"{stats['max_seq_length']}, grid asks for {params['max_seq_length']}"
            )
        cache.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, embeddings, allow_pickle=False)
        np.save(ids_path, item_ids, allow_pickle=False)
    return SemanticEmbeddingRecommender(min_rating=params["min_rating"]).fit(
        train, cand_set, embeddings_path=emb_path, item_ids_path=ids_path,
    )


BUILDERS = {"tfidf_cbf": build_tfidf, "embedding_cbf": build_embedding}


def append_log(row, path=LOG_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run(search_path, only=None):
    search = load_search(search_path)
    model = search["model"]
    if model not in BUILDERS:
        raise ValueError(f"No builder for model '{model}'")
    seed = search.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    combos = expand_grid(search["grid"])
    indices = range(len(combos)) if only is None else only
    train, _, cand_set, seen, relevant = build_protocol(str(ROOT / "data") + "/")
    k = search["fixed"].get("k", 10)
    commit, dirty = git_state()
    search_file = Path(search_path).resolve().relative_to(ROOT).as_posix()

    rows = []
    for idx in indices:
        params = {**search["fixed"], **combos[idx]}
        t0 = time.perf_counter()
        fitted = BUILDERS[model](params, train, cand_set)
        metrics = evaluate_model(fitted, relevant, seen, k=k)
        now = datetime.now()
        row = {
            "run_id": f"{model}_{now.strftime('%Y%m%d-%H%M%S')}_{idx:02d}",
            "timestamp": now.isoformat(timespec="seconds"),
            "model": model,
            "search_file": search_file,
            "combo_index": idx,
            "params": json.dumps(combos[idx], sort_keys=True),
            "commit": commit,
            "dirty": dirty,
            "selection_metric": search.get("selection_metric", ""),
            "Recall@10": metrics[f"Recall@{k}"],
            "NDCG@10": metrics[f"NDCG@{k}"],
            "HitRate@10": metrics[f"HitRate@{k}"],
            "n_users": metrics["n_users"],
            "seconds": round(time.perf_counter() - t0, 1),
        }
        append_log(row)
        rows.append(row)
        print(f"[{idx}] {row['params']} NDCG@10={row['NDCG@10']:.6f} "
              f"Recall@10={row['Recall@10']:.6f} ({row['seconds']}s)")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("search_file")
    parser.add_argument("--only", type=int, nargs="+",
                        help="run only these combination indices")
    parser.add_argument("--list", action="store_true",
                        help="print the combinations and exit")
    args = parser.parse_args(argv)

    combos = expand_grid(load_search(args.search_file)["grid"])
    if args.list:
        print(f"{len(combos)} combinations")
        for i, c in enumerate(combos):
            print(i, json.dumps(c))
        return
    run(args.search_file, args.only)


if __name__ == "__main__":
    main()
