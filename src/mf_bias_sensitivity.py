"""Pilot sensitivity analysis for MF item-bias weight and training epochs.

Run from the repository root with ``python -m src.mf_bias_sensitivity``.
The item-bias sweep fits once with W10 baseline parameters and changes only the
inference-time weight. The epoch sweep refits at 5 and 20 epochs. The same seeded
validation users are evaluated for every setting.
"""

import csv
import hashlib
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.matrix_factorization import MatrixFactorizationRecommender
from src.popularity import PopularityRecommender
from src.protocol import build_protocol, protocol_counts
from src.evaluate import recall_at_k, ndcg_at_k, hitrate_at_k


ROOT = Path(__file__).resolve().parents[1]
K = 10
SAMPLE_SIZE = 1000
SEED = 42
BIAS_WEIGHTS = (0.0, 0.25, 0.5, 1.0)
EPOCH_SETTINGS = (5, 20)
PARAMS = {
    "n_factors": 32,
    "n_epochs": 10,
    "learning_rate": 0.01,
    "regularization": 0.02,
    "min_rating": 4,
    "random_state": SEED,
    "item_bias_weight": 0.0,
}
CSV_PATH = ROOT / "outputs" / "metrics" / "mf_bias_sensitivity_pilot.csv"
LOG_PATH = ROOT / "outputs" / "logs" / "mf_bias_sensitivity_pilot.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    train, validation, candidates, seen, relevant = build_protocol(str(ROOT / "data") + "/")
    users = sorted(relevant, key=str)
    sample_users = random.Random(SEED).sample(users, min(SAMPLE_SIZE, len(users)))
    sample_relevant = {user: relevant[user] for user in sample_users}

    fit_start = time.perf_counter()
    mf = MatrixFactorizationRecommender(**PARAMS).fit(train, candidates)
    fit_seconds = time.perf_counter() - fit_start
    fit_seconds_by_epoch = {str(PARAMS["n_epochs"]): round(fit_seconds, 3)}

    popularity = PopularityRecommender(min_rating=4).fit(train, candidates)
    rows = []
    run_times = {}

    def score(name, model):
        start = time.perf_counter()
        totals = {"Recall@10": 0.0, "NDCG@10": 0.0, "HitRate@10": 0.0}
        unique_items = set()
        for user in sample_users:
            recs = model.recommend(user, seen, k=K)
            relevant_items = sample_relevant[user]
            totals["Recall@10"] += recall_at_k(recs, relevant_items, K)
            totals["NDCG@10"] += ndcg_at_k(recs, relevant_items, K)
            totals["HitRate@10"] += hitrate_at_k(recs, relevant_items, K)
            unique_items.update(recs)
        elapsed = time.perf_counter() - start
        run_times[name] = round(elapsed, 3)
        n_users = len(sample_users)
        rows.append({
            "model": name,
            "n_users": n_users,
            "Recall@10": totals["Recall@10"] / n_users,
            "NDCG@10": totals["NDCG@10"] / n_users,
            "HitRate@10": totals["HitRate@10"] / n_users,
            "unique_items_top10": len(unique_items),
            "catalog_coverage_pct": 100 * len(unique_items) / len(candidates),
        })

    score("Popularity", popularity)
    for weight in BIAS_WEIGHTS:
        mf.item_bias_weight = weight
        score(f"MF_item_bias_weight_{weight:g}", mf)

    for epochs in EPOCH_SETTINGS:
        epoch_params = {**PARAMS, "n_epochs": epochs, "item_bias_weight": 0.0}
        epoch_start = time.perf_counter()
        epoch_model = MatrixFactorizationRecommender(**epoch_params).fit(train, candidates)
        fit_seconds_by_epoch[str(epochs)] = round(time.perf_counter() - epoch_start, 3)
        score(f"MF_n_epochs_{epochs}_item_bias_weight_0", epoch_model)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    git_head = None
    try:
        import subprocess
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        pass

    shared_validation_metrics = {}
    reference_metrics_path = ROOT / "outputs" / "metrics" / "validation_results.csv"
    if reference_metrics_path.exists():
        with reference_metrics_path.open("r", newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                shared_validation_metrics[row["Model"]] = row
    reference_run_path = ROOT / "outputs" / "logs" / "validation_eval_run.json"
    shared_environment = None
    if reference_run_path.exists():
        shared_environment = json.loads(reference_run_path.read_text(encoding="utf-8")).get("environment")

    log = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "W10 pilot: compare inference-time MF item-bias weights and training epochs",
        "split": "validation",
        "k": K,
        "seed": SEED,
        "sample_size": len(sample_users),
        "sample_rule": "sorted evaluable-user IDs, then random.Random(42).sample",
        "protocol": protocol_counts(train, validation, candidates, seen, relevant),
        "mf_fit_params": PARAMS,
        "mf_scoring_item_bias_weights": list(BIAS_WEIGHTS),
        "mf_epoch_sensitivity_settings": list(EPOCH_SETTINGS),
        "interpretation_limit": "Pilot metrics use the same 1,000 validation users for all settings; not a full-population model selection result.",
        "fit_seconds": round(fit_seconds, 3),
        "fit_seconds_by_epoch_setting": fit_seconds_by_epoch,
        "evaluation_seconds_by_model": run_times,
        "results_csv": CSV_PATH.relative_to(ROOT).as_posix(),
        "shared_full_validation_reference": shared_validation_metrics,
        "shared_full_validation_environment": shared_environment,
        "cross_run_comparison_note": "Use the full-validation result file for official metrics. Absolute pilot metrics are not interchangeable because the pilot samples 1,000 users and uses a different Python/NumPy environment; compare only settings within this pilot.",
        "git_head": git_head,
        "input_sha256": {
            name: sha256(ROOT / "data" / name)
            for name in ["interactions_train.csv", "interactions_validation.csv", "candidate_items_warm.csv"]
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps({"results": rows, "fit_seconds": round(fit_seconds, 2), "log": LOG_PATH.relative_to(ROOT).as_posix()}, indent=2))


if __name__ == "__main__":
    main()
