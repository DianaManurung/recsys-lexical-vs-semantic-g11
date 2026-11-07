import math


def hits_and_positions(recs, relevant, k=10):
    """Posisi (mulai 1) dari item relevan di top-K rekomendasi."""
    return [i + 1 for i, item in enumerate(recs[:k]) if item in relevant]


def recall_at_k(recs, relevant, k=10):
    if not relevant:
        return 0.0
    return len(hits_and_positions(recs, relevant, k)) / len(relevant)


def ndcg_at_k(recs, relevant, k=10):
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(pos + 1) for pos in hits_and_positions(recs, relevant, k))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(pos + 1) for pos in range(1, ideal_hits + 1))
    return dcg / idcg


def hitrate_at_k(recs, relevant, k=10):
    return 1.0 if hits_and_positions(recs, relevant, k) else 0.0


def evaluate_model(model, relevant, seen, k=10):
    """Evaluasi satu model pada semua evaluable users.

    model       : objek dengan method recommend(user_id, seen, k)
    relevant    : {user_id: set(item relevan)}  (kunci = evaluable users)
    Mengembalikan rata-rata metrik dan jumlah user yang dievaluasi."""
    totals = {"Recall": 0.0, "NDCG": 0.0, "HitRate": 0.0}
    n = 0
    for user, rel in relevant.items():
        recs = model.recommend(user, seen, k)
        totals["Recall"] += recall_at_k(recs, rel, k)
        totals["NDCG"] += ndcg_at_k(recs, rel, k)
        totals["HitRate"] += hitrate_at_k(recs, rel, k)
        n += 1
    if n == 0:
        raise ValueError("Tidak ada evaluable users.")
    return {
        f"Recall@{k}": totals["Recall"] / n,
        f"NDCG@{k}": totals["NDCG"] / n,
        f"HitRate@{k}": totals["HitRate"] / n,
        "n_users": n,
    }


def evaluate_per_user(model, relevant, seen, k=10):
    """Seperti evaluate_model, tapi mengembalikan hasil PER USER (DataFrame).
    Dibutuhkan untuk analisis per kelompok user (RQ2) dan bootstrap."""
    import pandas as pd
    rows = []
    for user, rel in relevant.items():
        recs = model.recommend(user, seen, k)
        rows.append({
            "user_id": user,
            "n_relevant": len(rel),
            "recall": recall_at_k(recs, rel, k),
            "ndcg": ndcg_at_k(recs, rel, k),
            "hit": hitrate_at_k(recs, rel, k),
        })
    return pd.DataFrame(rows)


# Contoh kecil yang bisa dihitung manual: (nama, rekomendasi, relevan, k,
# Recall, NDCG, HitRate yang diharapkan)
MANUAL_CASES = [
    ("1 hit di posisi 2 dari 2 relevan", ["a", "b", "c"], {"b", "x"}, 3,
     0.5, (1 / math.log2(3)) / (1 + 1 / math.log2(3)), 1.0),
    ("hit di posisi 1, 1 relevan", ["a", "b", "c"], {"a"}, 3,
     1.0, 1.0, 1.0),
    ("tidak ada hit", ["a", "b", "c"], {"x", "y"}, 3,
     0.0, 0.0, 0.0),
    ("semua relevan di posisi teratas", ["a", "b", "c"], {"a", "b"}, 3,
     1.0, 1.0, 1.0),
    ("hit hanya di posisi 10", list("abcdefghij"), {"j"}, 10,
     1.0, 1 / math.log2(11), 1.0),
]


def sanity_check_metrics(verbose=True):
    """Cocokkan fungsi dengan hitungan manual. Melempar AssertionError jika beda."""
    for name, recs, rel, k, exp_recall, exp_ndcg, exp_hit in MANUAL_CASES:
        got = (recall_at_k(recs, rel, k), ndcg_at_k(recs, rel, k), hitrate_at_k(recs, rel, k))
        exp = (exp_recall, exp_ndcg, exp_hit)
        for g, e in zip(got, exp):
            assert abs(g - e) < 1e-9, f"{name}: dapat {got}, seharusnya {exp}"
        if verbose:
            print(f"OK  {name}: Recall={got[0]:.4f} NDCG={got[1]:.4f} HitRate={got[2]:.0f}")
    if verbose:
        print("Semua contoh manual cocok.")
    return True


if __name__ == "__main__":
    sanity_check_metrics()