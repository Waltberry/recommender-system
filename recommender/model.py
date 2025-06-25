import pandas as pd
from collections import defaultdict
import numpy as np
class PopularityRecommender:
    def __init__(self, events_df):
        self.events_df = events_df
        self.popular_items = None

    def fit(self, top_n=10, event_type="view"):
        # Get top N items by event_type
        event_data = self.events_df[self.events_df['event'] == event_type]
        self.popular_items = (
            event_data['itemid']
            .value_counts()
            .head(top_n)
            .index
            .tolist()
        )

    def recommend(self, user_id=None):
        return self.popular_items

class ItemItemRecommender:
    def __init__(self, events_df):
        self.events_df = events_df
        self.co_matrix = None
        self.item_index = {}
        self.index_item = {}

    def fit(self, min_item_freq=5):
        # Build co-occurrence matrix
        print("Building item-item co-occurrence matrix...")

        grouped = self.events_df.groupby("visitorid")["itemid"].apply(list)
        item_counts = defaultdict(int)
        co_counts = defaultdict(lambda: defaultdict(int))

        for item_list in grouped:
            unique_items = list(set(item_list))
            for i in range(len(unique_items)):
                item_i = unique_items[i]
                item_counts[item_i] += 1
                for j in range(i + 1, len(unique_items)):
                    item_j = unique_items[j]
                    co_counts[item_i][item_j] += 1
                    co_counts[item_j][item_i] += 1

        # Filter items by frequency
        valid_items = {k for k, v in item_counts.items() if v >= min_item_freq}
        self.item_index = {item: idx for idx, item in enumerate(valid_items)}
        self.index_item = {idx: item for item, idx in self.item_index.items()}

        size = len(self.item_index)
        self.co_matrix = np.zeros((size, size))

        for item_i, neighbors in co_counts.items():
            if item_i not in self.item_index:
                continue
            for item_j, score in neighbors.items():
                if item_j in self.item_index:
                    i, j = self.item_index[item_i], self.item_index[item_j]
                    self.co_matrix[i, j] = score

        print("Matrix built successfully.")

    def recommend(self, item_id, top_n=5):
        if item_id not in self.item_index:
            return []

        idx = self.item_index[item_id]
        similarities = self.co_matrix[idx]
        top_indices = np.argsort(similarities)[::-1][:top_n + 1]
        recommended_ids = [self.index_item[i] for i in top_indices if i != idx]
        return recommended_ids[:top_n]


class CartAwareRecommender:
    def __init__(self, events_df):
        self.events_df = events_df
        self.item_sim_recommender = ItemItemRecommender(events_df)
        self.pop_recommender = PopularityRecommender(events_df)

    def fit(self):
        self.item_sim_recommender.fit(min_item_freq=10)
        self.pop_recommender.fit(top_n=10, event_type="transaction")

    def recommend(self, cart_items, top_n=5):
        # Collect recommendations from item similarity
        recommendations = []
        seen = set(cart_items)

        for item in cart_items:
            recs = self.item_sim_recommender.recommend(item, top_n=top_n)
            for r in recs:
                if r not in seen:
                    recommendations.append(r)
                    seen.add(r)

        # If not enough, fallback to popular items
        if len(recommendations) < top_n:
            for pop_item in self.pop_recommender.recommend():
                if pop_item not in seen:
                    recommendations.append(pop_item)
                    seen.add(pop_item)
                if len(recommendations) >= top_n:
                    break

        return recommendations[:top_n]


class UnifiedRecommender:
    def __init__(self, events_df):
        self.cart_recommender = CartAwareRecommender(events_df)

    def fit(self):
        self.cart_recommender.fit()

    def recommend(self, cart_items=[], top_n=5):
        return self.cart_recommender.recommend(cart_items, top_n)
