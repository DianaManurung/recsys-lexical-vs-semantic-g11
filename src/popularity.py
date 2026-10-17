class PopularityRecommender:
    def __init__(self, min_rating=4):
        self.min_rating = min_rating
        self.ranked_items = []

    def fit(self, train, cand_set):
        pos = train[train["rating"] >= self.min_rating]
        pop = pos["item_id"].value_counts()
        pop = pop[pop.index.isin(cand_set)]
        self.ranked_items = pop.index.tolist()
        return self

    def recommend(self, user_id, seen, k=10):
        sudah = seen.get(user_id, set())
        out = []
        for item in self.ranked_items:
            if item not in sudah:
                out.append(item)
                if len(out) == k:
                    break
        return out
