from pathlib import Path

# Nama kolom di CSV. SESUAIKAN kalau berbeda (cek dengan: print(train.columns))
USER_COL = "user_id"
ITEM_COL = "item_id"
TIME_COL = "timestamp"

FORBIDDEN_FEATURES = {"average_rating", "rating_number"}
# Dirangkai supaya file ini tidak "menemukan dirinya sendiri" saat scan
TEST_MARKERS = ("interactions_" + "test", "test" + "_interactions")


class LeakageError(AssertionError):
    """Dilempar kalau ada indikasi data leakage."""


def check_temporal_split(train, val, time_col=TIME_COL):
    """Aturan 2: semua interaksi training harus lebih lama dari validation."""
    if train[time_col].max() > val[time_col].min():
        raise LeakageError(
            "Split temporal rusak: ada interaksi training yang lebih baru "
            "daripada interaksi validation.")
    return True


def check_no_pair_overlap(train, val, user_col=USER_COL, item_col=ITEM_COL):
    """Aturan 1-2: pasangan (user, item) validation tidak boleh ada di training."""
    t = set(zip(train[user_col], train[item_col]))
    v = set(zip(val[user_col], val[item_col]))
    overlap = t & v
    if overlap:
        raise LeakageError(
            f"{len(overlap)} pasangan (user, item) muncul di training dan validation.")
    return True


def check_features(feature_columns):
    """Aturan 3: fitur agregat crawl-time tidak boleh dipakai."""
    bad = FORBIDDEN_FEATURES & set(feature_columns)
    if bad:
        raise LeakageError(f"Fitur terlarang dipakai: {sorted(bad)}")
    return True


def check_relevant_in_candidates(relevant, cand_set):
    """Item relevan validation harus termasuk kandidat warm (protokol bersama)."""
    outside = {i for items in relevant.values() for i in items if i not in cand_set}
    if outside:
        raise LeakageError(
            f"{len(outside)} item relevan berada di luar candidate set warm.")
    return True


def check_recommendations(recs_by_user, cand_set, seen):
    """Aturan 4: rekomendasi harus dari candidate set dan bukan item yang
    sudah dilihat user di training. recs_by_user: {user_id: [item, ...]}"""
    for user, recs in recs_by_user.items():
        outside = [i for i in recs if i not in cand_set]
        if outside:
            raise LeakageError(
                f"User {user}: {len(outside)} item di luar candidate set, "
                f"misalnya {outside[:3]}")
        already = [i for i in recs if i in seen.get(user, set())]
        if already:
            raise LeakageError(
                f"User {user}: merekomendasikan item yang sudah ada di "
                f"training, misalnya {already[:3]}")
    return True


def check_same_candidates(candidate_sets):
    """Aturan 4: semua model memakai candidate universe yang sama.
    candidate_sets: {"popularity": set(...), "mf": set(...), ...}"""
    names = list(candidate_sets)
    base = candidate_sets[names[0]]
    for name in names[1:]:
        if candidate_sets[name] != base:
            raise LeakageError(
                f"Candidate set '{name}' berbeda dari '{names[0]}'.")
    return True


def check_test_unused(root=".", folders=("src", "notebooks")):
    """Aturan 5: tidak ada kode/notebook yang membaca file test pada CP3."""
    hits = []
    for folder in folders:
        base = Path(root) / folder
        if not base.exists():
            continue
        for path in list(base.rglob("*.py")) + list(base.rglob("*.ipynb")):
            if path.name == Path(__file__).name:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(m in text for m in TEST_MARKERS):
                hits.append(str(path))
    if hits:
        raise LeakageError(f"File test disebut di: {hits}")
    return True


def run_static_checks(train, val, relevant, cand_set, root="."):
    """Cek yang tidak butuh model: split, overlap, kandidat, test."""
    results = {
        "temporal_split": check_temporal_split(train, val),
        "no_pair_overlap": check_no_pair_overlap(train, val),
        "relevant_in_candidates": check_relevant_in_candidates(relevant, cand_set),
        "test_unused": check_test_unused(root),
    }
    return results


if __name__ == "__main__":
    # Contoh pemakaian pada data nyata + baseline Popularity
    from src.protocol import build_protocol
    from src.popularity import PopularityRecommender

    train, val, cand_set, seen, relevant = build_protocol("data/")
    print("Cek statis:", run_static_checks(train, val, relevant, cand_set))

    pop = PopularityRecommender().fit(train, cand_set)
    sample_users = list(relevant)[:500]
    recs = {u: pop.recommend(u, seen, 10) for u in sample_users}
    print("Cek rekomendasi Popularity (500 user):",
          check_recommendations(recs, cand_set, seen))
    print("Semua pemeriksaan leakage lolos.")
