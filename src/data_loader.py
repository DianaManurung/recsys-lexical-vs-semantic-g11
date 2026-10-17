import pandas as pd

def load_data(data_dir="data/"):
    train = pd.read_csv(data_dir + "interactions_train.csv")
    val = pd.read_csv(data_dir + "interactions_validation.csv")
    cands = pd.read_csv(data_dir + "candidate_items_warm.csv")["item_id"]
    cand_set = set(cands)
    # item yang sudah dilihat tiap user di TRAIN (semua rating)
    seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
    return train, val, cand_set, seen

def relevant_items_val(val, cand_set, seen, min_rating=4):
    """Item relevan per user di validation (rating >= 4)."""
    v = val[(val["rating"] >= min_rating)
            & val["item_id"].isin(cand_set)
            & val["user_id"].isin(seen.keys())]
    return v.groupby("user_id")["item_id"].apply(set).to_dict()
