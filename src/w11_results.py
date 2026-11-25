"""W11 validation results: best configuration per experiment, K = 10/20/50,
baseline comparison, RQ2 user groups, paired bootstrap, tables and figures.

The best configuration of E2-E6 is read from outputs/logs/experiment_log.csv
with the selection rule of configs/reproducibility.json (highest NDCG@10,
then Recall@10, then lower combination index; evaluable_users = official).
Each best configuration is rebuilt with the same builder as src/run_grid.py
and evaluated once more on validation at K = 10, 20 and 50. The test split is
never read.

Usage (from the repository root, after the grids in configs/*_search.json
have been run with src/run_grid.py):
    python -m src.w11_results
"""

import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate_per_user_at_ks, summarize_at_ks  # noqa: E402
from src.leakage_check import check_recommendations  # noqa: E402
from src.popularity import PopularityRecommender  # noqa: E402
from src.run_grid import BUILDERS, best_run, git_state, read_log  # noqa: E402
from src.shared_eval import build_protocol, check_top_k, evaluable_users_rule  # noqa: E402
from src.user_groups import assign_groups, load_group_config, train_activity  # noqa: E402

KS = (10, 20, 50)
SEED = 42
N_BOOT = 1000
METRICS_DIR = ROOT / "outputs" / "metrics"
FIG_DIR = ROOT / "outputs" / "figures"
LOG_DIR = ROOT / "outputs" / "logs"

# (label, experiment id, search file). Order fixes table rows and colors.
EXPERIMENTS = [
    ("Popularity", "E1", None),
    ("MF", "E2", "configs/mf_search.json"),
    ("TF-IDF CBF", "E3", "configs/tfidf_search.json"),
    ("Embedding CBF", "E4", "configs/embedding_search.json"),
    ("Hybrid TF-IDF + Pop", "E5", "configs/hybrid_search.json"),
    ("Hybrid Embedding + Pop", "E6", "configs/hybrid_embedding_search.json"),
]
# Categorical slots 1-6 of the dataviz reference palette (validated, light).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


class Recording:
    """Keep each user's top-10 for the leakage checks."""

    def __init__(self, model):
        self.model, self.top10 = model, {}

    def recommend(self, user_id, seen, k=10):
        recs = self.model.recommend(user_id, seen, k)
        self.top10[user_id] = recs[:10]
        return recs


def environment():
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for lib in ("numpy", "pandas", "scipy", "scikit-learn", "sentence-transformers",
                "torch", "matplotlib"):
        try:
            env[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            env[lib] = None
    return env


def build_models(train, cand_set, rule):
    """Return [(label, exp, run_row or None, fitted model, seconds)]."""
    out = []
    for label, exp, search in EXPERIMENTS:
        t0 = time.perf_counter()
        if search is None:
            row = None
            model = PopularityRecommender(min_rating=4).fit(train, cand_set)
        else:
            row = best_run(search, rule)
            if row is None:
                raise RuntimeError(f"No logged runs for {search}; run src/run_grid.py first")
            params = json.loads(row["full_params"])
            model = BUILDERS[row["model"]](params, train, cand_set)
        out.append((label, exp, row, model, round(time.perf_counter() - t0, 1)))
        print(f"built {exp} {label} ({out[-1][4]}s)", flush=True)
    return out


def paired_bootstrap(a, b, rng, n_boot=N_BOOT):
    """Mean difference b - a with a 95% percentile interval over users."""
    a, b = np.asarray(a), np.asarray(b)
    d = b - a
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff": float(d.mean()),
            "rel_diff_pct": float(100 * d.mean() / a.mean()) if a.mean() else None,
            "ci95": [float(lo), float(hi)],
            "significant": bool(lo > 0 or hi < 0),
            "p_two_sided": float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))}


def hparam_table(rule):
    """All logged runs (official users) of the W11 search files, ranked."""
    rows = [r for r in read_log() if r.get("evaluable_users") == rule
            and r["search_file"] in {s for _, _, s in EXPERIMENTS if s}]
    df = pd.DataFrame(rows)
    for c in ("Recall@10", "NDCG@10", "HitRate@10", "seconds"):
        df[c] = df[c].astype(float)
    df["combo_index"] = df["combo_index"].astype(int)
    # A combination that was logged more than once keeps its latest run.
    df = df.sort_values("timestamp").drop_duplicates(["search_file", "combo_index"], keep="last")
    exp_of = {s: e for _, e, s in EXPERIMENTS if s}
    df["experiment"] = df["search_file"].map(exp_of)
    df = df.sort_values(["experiment", "NDCG@10", "Recall@10", "combo_index"],
                        ascending=[True, False, False, True])
    df["rank"] = df.groupby("experiment").cumcount() + 1
    return df[["experiment", "rank", "run_id", "combo_index", "params", "Recall@10",
               "NDCG@10", "HitRate@10", "seconds", "commit", "timestamp"]]


# ----------------------------------------------------------------- figures
def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor=SURFACE)


def make_figures(results, groups_df, hp, alpha_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = [e[0] for e in EXPERIMENTS]

    # 1. Grouped bar: NDCG@K by model, grouped by K.
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    width = 0.8 / len(labels)
    x = np.arange(len(KS))
    for j, lab in enumerate(labels):
        vals = [results.loc[lab, f"NDCG@{k}"] for k in KS]
        ax.bar(x + (j - (len(labels) - 1) / 2) * width, vals, width * 0.92,
               color=COLORS[j], label=lab, edgecolor=SURFACE, linewidth=0.8)
    ax.set_xticks(x, [f"K = {k}" for k in KS])
    _style(ax, "NDCG@K per model (validation, 11,659 users)", "Cutoff K", "NDCG@K")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    _save(fig, "w11_ndcg_by_model"); plt.close(fig)

    # 2. Line chart: Recall@K and NDCG@K vs K (two panels, one y-axis each).
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    for ax, metric in zip(axes, ("Recall", "NDCG")):
        for j, lab in enumerate(labels):
            ax.plot(KS, [results.loc[lab, f"{metric}@{k}"] for k in KS], color=COLORS[j],
                    linewidth=2, marker="o", markersize=5, label=lab)
        ax.set_xticks(KS)
        _style(ax, f"{metric}@K vs K", "Cutoff K", f"{metric}@K")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    _save(fig, "w11_metrics_vs_k"); plt.close(fig)

    # 3. Heatmaps of the E3 and E2 search spaces (NDCG@10).
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)

    def heat(exp, row_keys, col_keys, title, name):
        sub = hp[hp.experiment == exp].copy()
        p = sub["params"].map(json.loads)
        sub["row"] = p.map(lambda d: ", ".join(f"{k}={d[k]}" for k in row_keys))
        sub["col"] = p.map(lambda d: ", ".join(f"{k}={d[k]}" for k in col_keys))
        table = sub.pivot(index="row", columns="col", values="NDCG@10")
        fig, ax = plt.subplots(figsize=(1.9 * table.shape[1] + 2.5, 0.75 * table.shape[0] + 1.6),
                               facecolor=SURFACE)
        im = ax.imshow(table.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(table.shape[1]), table.columns, rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(table.shape[0]), table.index, fontsize=8)
        vmid = np.nanmean(table.values)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                v = table.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=8,
                            color="#ffffff" if v > vmid else INK)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.04)
        cb.set_label("NDCG@10", color=INK2)
        cb.outline.set_visible(False)
        for s in ax.spines.values():
            s.set_visible(False)
        _save(fig, name); plt.close(fig)

    heat("E3", ["max_features"], ["ngram_range", "min_df"],
         "E3 TF-IDF search space: NDCG@10", "w11_tfidf_grid_heatmap")
    heat("E2", ["n_factors", "regularization"], ["learning_rate", "n_epochs"],
         "E2 MF search space: NDCG@10", "w11_mf_grid_heatmap")

    # 4. Hybrid alpha: contribution of content vs popularity.
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    for lab, color in (("Hybrid TF-IDF + Pop", COLORS[4]), ("Hybrid Embedding + Pop", COLORS[5])):
        d = alpha_df[alpha_df.model == lab].sort_values("alpha")
        ax.plot(d.alpha, d["NDCG@10"], color=color, linewidth=2, marker="o", markersize=6, label=lab)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0],
                  ["0\n(popularity\nonly)", "0.25", "0.5", "0.75", "1\n(content\nonly)"])
    _style(ax, "Hybrid weight alpha vs NDCG@10", "alpha (content weight)", "NDCG@10")
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    _save(fig, "w11_hybrid_alpha"); plt.close(fig)

    # 5. RQ2: NDCG@10 per activity group with 95% bootstrap intervals.
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURFACE)
    show = [("TF-IDF CBF", COLORS[2]), ("Embedding CBF", COLORS[3])]
    order = list(groups_df["group"].cat.categories) + ["all"]
    x = np.arange(len(order))
    w = 0.36
    for j, (lab, color) in enumerate(show):
        d = groups_df[groups_df.model == lab].set_index("group_str").loc[order]
        y = d["NDCG@10"].values
        err = np.vstack([y - d["ci_lo"].values, d["ci_hi"].values - y])
        ax.bar(x + (j - 0.5) * w, y, w * 0.92, color=color, label=lab,
               edgecolor=SURFACE, linewidth=0.8)
        ax.errorbar(x + (j - 0.5) * w, y, yerr=err, fmt="none", ecolor=INK2,
                    elinewidth=1, capsize=3)
    n_by = groups_df[groups_df.model == show[0][0]].set_index("group_str").loc[order, "n_users"]
    ax.set_xticks(x, [f"{g}\n(n={n:,})" for g, n in zip(order, n_by)])
    _style(ax, "RQ2: TF-IDF vs Embedding by train activity (NDCG@10, 95% CI)",
           "User group (train interactions: rendah <=5, sedang 6-8, tinggi >=9)", "NDCG@10")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "w11_rq2_groups"); plt.close(fig)


# -------------------------------------------------------------------- main
def run():
    t_start = time.perf_counter()
    rule = evaluable_users_rule()
    train, _, cand_set, seen, relevant = build_protocol(str(ROOT / "data") + "/")
    users = list(relevant)
    commit, dirty = git_state()

    built = build_models(train, cand_set, rule)
    per_user, rows, leakage, consistency, run_info = {}, [], {}, {}, {}
    for label, exp, row, model, secs in built:
        t0 = time.perf_counter()
        rec = Recording(model)
        pu = evaluate_per_user_at_ks(rec, relevant, seen, KS)
        per_user[label] = pu
        summary = summarize_at_ks(pu, KS)
        check_recommendations(rec.top10, cand_set, seen)
        leakage[label] = check_top_k(rec.top10, cand_set, seen, k=10)
        if row is not None:
            consistency[label] = {
                "logged_NDCG@10": float(row["NDCG@10"]),
                "recomputed_NDCG@10": summary["NDCG@10"],
                "abs_diff": abs(float(row["NDCG@10"]) - summary["NDCG@10"]),
            }
        run_info[label] = {
            "experiment": exp,
            "run_id": row["run_id"] if row else None,
            "combo_index": int(row["combo_index"]) if row else None,
            "params": json.loads(row["full_params"]) if row else {"min_rating": 4},
            "build_seconds": secs,
            "eval_seconds": round(time.perf_counter() - t0, 1),
            "unique_items_top10": len({i for r in rec.top10.values() for i in r}),
        }
        rows.append({"experiment": exp, "model": label, **summary,
                     "run_id": run_info[label]["run_id"],
                     "params": json.dumps({k: v for k, v in run_info[label]["params"].items()
                                           if k not in ("k",)}, sort_keys=True)})
        print(f"{exp} {label}: NDCG@10={summary['NDCG@10']:.6f} Recall@10={summary['Recall@10']:.6f}",
              flush=True)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(METRICS_DIR / "w11_validation_results.csv", index=False)
    res_idx = results.set_index("model")

    # Baseline comparison: every model against E1, E2 and E3 (absolute and relative).
    comp = []
    for base in ("Popularity", "MF", "TF-IDF CBF"):
        for label, *_ in EXPERIMENTS:
            if label == base:
                continue
            for m in ("NDCG@10", "Recall@10", "NDCG@20", "Recall@20"):
                b, v = res_idx.loc[base, m], res_idx.loc[label, m]
                comp.append({"model": label, "baseline": base, "metric": m, "baseline_value": b,
                             "model_value": v, "abs_diff": v - b,
                             "rel_diff_pct": 100 * (v - b) / b if b else None})
    pd.DataFrame(comp).to_csv(METRICS_DIR / "w11_baseline_comparison.csv", index=False)

    # Per-user table and RQ2 groups.
    cfg = load_group_config()
    activity = train_activity(train).reindex(users)
    groups = pd.Series(assign_groups(activity, cfg), index=users)
    wide = pd.DataFrame({"user_id": users, "group": groups.values.astype(str),
                         "train_interactions": activity.values})
    for label, pu in per_user.items():
        for c in ("recall@10", "ndcg@10", "recall@20", "ndcg@20"):
            wide[f"{label}|{c}"] = pu[c].values
    wide.to_csv(METRICS_DIR / "w11_per_user_val.csv", index=False)

    rng = np.random.default_rng(SEED)
    grows = []
    for label, pu in per_user.items():
        for g in list(groups.cat.categories) + ["all"]:
            mask = np.ones(len(users), bool) if g == "all" else (groups.values == g)
            sub = pu[mask]
            s = summarize_at_ks(sub, (10,))
            boots = sub["ndcg@10"].values[rng.integers(0, len(sub), (N_BOOT, len(sub)))].mean(1)
            grows.append({"model": label, "group": g, "n_users": int(mask.sum()),
                          "Recall@10": s["Recall@10"], "NDCG@10": s["NDCG@10"],
                          "HitRate@10": s["HitRate@10"],
                          "ci_lo": float(np.percentile(boots, 2.5)),
                          "ci_hi": float(np.percentile(boots, 97.5))})
    groups_df = pd.DataFrame(grows)
    groups_df.to_csv(METRICS_DIR / "w11_user_groups.csv", index=False)
    groups_df["group_str"] = groups_df["group"]
    groups_df["group"] = pd.Categorical(groups_df["group"],
                                        categories=list(groups.cat.categories))

    # Paired bootstrap tests (per-user NDCG@10 and Recall@10).
    pairs = [("TF-IDF CBF", "Embedding CBF"), ("Hybrid TF-IDF + Pop", "Hybrid Embedding + Pop"),
             ("Popularity", "Hybrid TF-IDF + Pop"), ("TF-IDF CBF", "Hybrid TF-IDF + Pop"),
             ("Popularity", "Hybrid Embedding + Pop"), ("Embedding CBF", "Hybrid Embedding + Pop"),
             ("Popularity", "TF-IDF CBF"), ("MF", "Popularity")]
    sig = {"method": f"paired bootstrap over users, {N_BOOT} resamples, seed {SEED}; diff = b - a",
           "overall": [], "rq2_by_group": []}
    for a, b in pairs:
        for m in ("ndcg@10", "recall@10"):
            sig["overall"].append({"a": a, "b": b, "metric": m,
                                   **paired_bootstrap(per_user[a][m], per_user[b][m], rng)})
    for g in groups.cat.categories:
        mask = groups.values == g
        for m in ("ndcg@10", "recall@10"):
            sig["rq2_by_group"].append({
                "group": g, "n_users": int(mask.sum()), "a": "TF-IDF CBF", "b": "Embedding CBF",
                "metric": m, **paired_bootstrap(per_user["TF-IDF CBF"][m][mask],
                                                per_user["Embedding CBF"][m][mask], rng)})
    (METRICS_DIR / "w11_significance.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")

    # Hyperparameter search table.
    hp = hparam_table(rule)
    hp.to_csv(METRICS_DIR / "w11_hparam_search.csv", index=False)

    # Hybrid alpha curve: grid points from the log; alpha = 0 and 1 are the
    # pure components (popularity; content model with the hybrid's fixed params).
    from src.content_based import TfidfContentRecommender
    from src.evaluate import evaluate_model
    from src.semantic_based import SemanticEmbeddingRecommender
    pure_tfidf = TfidfContentRecommender(stop_words="english", min_df=2, max_df=0.8,
                                         sublinear_tf=True, min_rating=4).fit(train, cand_set)
    pure_emb = SemanticEmbeddingRecommender(min_rating=4).fit(train, cand_set)
    pop_ndcg = res_idx.loc["Popularity", "NDCG@10"]
    alpha_rows = []
    for lab, exp, pure in (("Hybrid TF-IDF + Pop", "E5", pure_tfidf),
                           ("Hybrid Embedding + Pop", "E6", pure_emb)):
        alpha_rows.append({"model": lab, "alpha": 0.0, "NDCG@10": pop_ndcg, "source": "E1 Popularity"})
        for _, r in hp[hp.experiment == exp].iterrows():
            alpha_rows.append({"model": lab, "alpha": json.loads(r["params"])["alpha"],
                               "NDCG@10": r["NDCG@10"], "source": r["run_id"]})
        alpha_rows.append({"model": lab, "alpha": 1.0,
                           "NDCG@10": evaluate_model(pure, relevant, seen, 10)["NDCG@10"],
                           "source": "pure content model with the hybrid's fixed parameters"})
    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df.to_csv(METRICS_DIR / "w11_hybrid_alpha.csv", index=False)

    make_figures(res_idx, groups_df, hp, alpha_df)

    log = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "src/w11_results.py",
        "commit": commit, "dirty": dirty, "seed": SEED, "ks": list(KS),
        "evaluable_users_rule": rule, "n_users": len(users),
        "selection_rule": "configs/reproducibility.json (highest NDCG@10, then Recall@10, then lower combo index)",
        "best_runs": run_info,
        "log_vs_recomputed_NDCG@10": consistency,
        "leakage_top10": leakage,
        "test_split_used": False,
        "environment": environment(),
        "seconds_total": round(time.perf_counter() - t_start, 1),
    }
    (LOG_DIR / "w11_results_run.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(consistency, indent=1))
    print(json.dumps(leakage))
    print("done in", log["seconds_total"], "s")


if __name__ == "__main__":
    run()
