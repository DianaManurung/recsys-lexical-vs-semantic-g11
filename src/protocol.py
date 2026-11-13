import json
import os

from src.data_loader import load_data, relevant_items_val

MIN_RATING = 4   
TOP_K = 10       
OFFICIAL_FILE = "evaluable_users_validation_warm.csv"


def build_protocol(data_dir="data/", min_rating=MIN_RATING):
    """Kembalikan (train, val, cand_set, seen, relevant).

    Evaluable users (aturan resmi paket proyek): user punya minimal 1 item
    disukai (rating >= min_rating) di training DAN minimal 1 di validation.
    `seen` tetap berisi SEMUA item training (semua rating) supaya item yang
    pernah diinteraksikan user tidak direkomendasikan lagi."""
    train, val, cand_set, seen = load_data(data_dir)
    relevant = relevant_items_val(val, cand_set, seen, min_rating)
    positive_train_users = set(train.loc[train["rating"] >= min_rating, "user_id"])
    relevant = {u: items for u, items in relevant.items() if u in positive_train_users}
    return train, val, cand_set, seen, relevant


def protocol_counts(train, val, cand_set, seen, relevant, min_rating=MIN_RATING):
    """Ringkasan angka protokol untuk log dan laporan."""
    train_items = set(train["item_id"])
    return {
        "train_interactions": int(len(train)),
        "validation_interactions": int(len(val)),
        "train_users": int(len(seen)),
        "train_items": int(len(train_items)),
        "candidate_items_warm": int(len(cand_set)),
        "candidates_not_seen_in_train": int(len(cand_set - train_items)),
        "evaluable_users": int(len(relevant)),
        "relevant_pairs": int(sum(len(v) for v in relevant.values())),
        "min_rating_relevant": min_rating,
        "top_k": TOP_K,
    }


def compare_with_official(relevant, data_dir="data/"):
    """Bandingkan evaluable users kita dengan file resmi dosen (jika ada)."""
    import pandas as pd
    path = os.path.join(data_dir, OFFICIAL_FILE)
    if not os.path.exists(path):
        return {"official_file_found": False,
                "note": f"{OFFICIAL_FILE} belum ada di {data_dir}"}
    df = pd.read_csv(path)
    col = "user_id" if "user_id" in df.columns else df.columns[0]
    official = set(df[col])
    ours = set(relevant)
    return {
        "official_file_found": True,
        "official_count": len(official),
        "our_count": len(ours),
        "identical": official == ours,
        "only_in_official": len(official - ours),
        "only_in_ours": len(ours - official),
    }


def save_log(counts, path="outputs/logs/protocol_counts.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    return path


if __name__ == "__main__":
    train, val, cand_set, seen, relevant = build_protocol("data/")
    counts = protocol_counts(train, val, cand_set, seen, relevant)
    counts["check_vs_official"] = compare_with_official(relevant, "data/")
    for key, value in counts.items():
        print(f"{key}: {value}")
    print("Log disimpan di:", save_log(counts))