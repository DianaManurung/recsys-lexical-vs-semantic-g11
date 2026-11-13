"""Grid search runner for ``configs/*_search.json`` (validation only).

Every combination of the ``grid`` section is merged with ``fixed``, fitted on
train, evaluated on validation with ``src.evaluate.evaluate_model`` (through
``src/shared_eval.py``), and appended as one row to
``outputs/logs/experiment_log.csv``. Run IDs, logged fields and the selection
rule follow ``configs/reproducibility.json``.

Usage (from the repository root):
    python -m src.run_grid configs/tfidf_search.json --list
    python -m src.run_grid configs/tfidf_search.json --only 0 1
    python -m src.run_grid configs/tfidf_search.json --best
    python -m src.run_grid configs/tfidf_search.json          # full grid
"""

import argparse
import csv
import hashlib
import itertools
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared_eval import build_protocol, evaluable_users_rule, evaluate_model  # noqa: E402

LOG_PATH = ROOT / "outputs" / "logs" / "experiment_log.csv"
# Columns from the CP3 log are kept; the rest were added in W10 to cover the
# fields listed in configs/reproducibility.json. Older rows leave them empty.
LOG_COLUMNS = [
    "run_id", "timestamp", "model", "search_file", "combo_index", "params",
    "commit", "dirty", "selection_metric", "Recall@10", "NDCG@10",
    "HitRate@10", "n_users", "seconds",
    "seed", "evaluable_users", "full_params", "data_sha256", "environment",
]
DATA_FILES = ["interactions_train.csv", "interactions_validation.csv",
              "candidate_items_warm.csv", "items.csv"]
LIBRARIES = ["numpy", "pandas", "scipy", "scikit-learn",
             "sentence-transformers", "torch"]


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


def data_fingerprints():
    """SHA-256 of the input CSV files (train, validation, candidates, items)."""
    out = {}
    for name in DATA_FILES:
        digest = hashlib.sha256()
        with open(ROOT / "data" / name, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        out[name] = digest.hexdigest()
    return out


def environment():
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for lib in LIBRARIES:
        try:
            env[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            env[lib] = None
    return env


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


def read_log(path=LOG_PATH):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_log(row, path=LOG_PATH):
    """Append one row. A log with an older header is rewritten with the
    current columns; existing rows are kept and new columns left empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_log(path)
    header = list(rows[0]) if rows else None
    if rows and header != LOG_COLUMNS:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, restval="")
            writer.writeheader()
            writer.writerows(rows)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if not rows:
            writer.writeheader()
        writer.writerow(row)


def best_run(search_file, rule, path=LOG_PATH):
    """Selection rule of configs/reproducibility.json among logged runs of one
    search file and evaluable-user rule: highest NDCG@10, then higher
    Recall@10, then lower combination index."""
    rows = [r for r in read_log(path)
            if r["search_file"] == search_file and r.get("evaluable_users") == rule]
    if not rows:
        return None
    return min(rows, key=lambda r: (-float(r["NDCG@10"]), -float(r["Recall@10"]),
                                    int(r["combo_index"])))


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
    rule = evaluable_users_rule()
    train, _, cand_set, seen, relevant = build_protocol(str(ROOT / "data") + "/")
    k = search["fixed"].get("k", 10)
    commit, dirty = git_state()
    search_file = Path(search_path).resolve().relative_to(ROOT).as_posix()
    fingerprints = json.dumps(data_fingerprints(), sort_keys=True)
    env = json.dumps(environment(), sort_keys=True)

    rows = []
    for idx in indices:
        params = {**search["fixed"], **combos[idx]}
        t0 = time.perf_counter()
        fitted = BUILDERS[model](params, train, cand_set)
        metrics = evaluate_model(fitted, relevant, seen, k=k)
        now = datetime.now(timezone.utc)
        row = {
            "run_id": f"{model}_{now.strftime('%Y%m%d-%H%M%S')}_{idx}_s{seed}",
            "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
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
            "seed": seed,
            "evaluable_users": rule,
            "full_params": json.dumps(params, sort_keys=True),
            "data_sha256": fingerprints,
            "environment": env,
        }
        append_log(row)
        rows.append(row)
        print(f"[{idx}] {row['params']} NDCG@10={row['NDCG@10']:.6f} "
              f"Recall@10={row['Recall@10']:.6f} n={row['n_users']} ({row['seconds']}s)")
    return rows


def print_best(search_path):
    search_file = Path(search_path).resolve().relative_to(ROOT).as_posix()
    rule = evaluable_users_rule()
    best = best_run(search_file, rule)
    if best is None:
        print(f"No logged runs for {search_file} with evaluable_users={rule}")
        return None
    print(f"Best logged run so far ({rule}): {best['run_id']} combo {best['combo_index']} "
          f"{best['params']} NDCG@10={float(best['NDCG@10']):.6f} "
          f"Recall@10={float(best['Recall@10']):.6f}")
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("search_file")
    parser.add_argument("--only", type=int, nargs="+",
                        help="run only these combination indices")
    parser.add_argument("--list", action="store_true",
                        help="print the combinations and exit")
    parser.add_argument("--best", action="store_true",
                        help="print the best logged run and exit")
    args = parser.parse_args(argv)

    combos = expand_grid(load_search(args.search_file)["grid"])
    if args.list:
        print(f"{len(combos)} combinations")
        for i, c in enumerate(combos):
            print(i, json.dumps(c))
        return
    if args.best:
        print_best(args.search_file)
        return
    run(args.search_file, args.only)
    print_best(args.search_file)


if __name__ == "__main__":
    main()
