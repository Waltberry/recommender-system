import pandas as pd
from recommender.model import (
    PopularityRecommender,
    ItemItemRecommender,
    CartAwareRecommender,
    UnifiedRecommender
)

def load_data(path="data/processed/events_cleaned.csv"):
    print(f"Loading data from {path}...")
    events_df = pd.read_csv(path)
    print(f"Data loaded: {len(events_df)} rows.")
    return events_df

def test_popularity_recommender(events):
    print("\n--- Testing PopularityRecommender ---")
    model = PopularityRecommender(events)
    model.fit(top_n=10, event_type="transaction")
    recommendations = model.recommend()
    print(f"Top Popular Items: {recommendations}")

def test_item_item_recommender(events):
    print("\n--- Testing ItemItemRecommender ---")
    model = ItemItemRecommender(events)
    model.fit(min_item_freq=10)
    test_item = events['itemid'].value_counts().idxmax()
    print(f"Most Frequent Item: {test_item}")
    recommendations = model.recommend(test_item, top_n=5)
    print(f"Recommended Items for '{test_item}': {recommendations}")

def test_cart_aware_recommender(events):
    print("\n--- Testing CartAwareRecommender ---")
    model = CartAwareRecommender(events)
    model.fit()
    sample_cart = events.query("event == 'addtocart'")['itemid'].head(3).tolist()
    print(f"Sample Cart Items: {sample_cart}")
    recommendations = model.recommend(sample_cart, top_n=5)
    print(f"Cart-aware Recommendations: {recommendations}")

def test_unified_recommender(events):
    print("\n--- Testing UnifiedRecommender ---")
    model = UnifiedRecommender(events)
    model.fit()
    sample_cart = events.query("event == 'addtocart'")['itemid'].head(3).tolist()
    if not sample_cart:
        print("Cart is empty — cannot generate recommendations.")
        return
    recommendations = model.recommend(cart_items=sample_cart, top_n=5)
    print(f"Unified Recommender Recommendations: {recommendations}")

def main():
    events = load_data()
    test_popularity_recommender(events)
    test_item_item_recommender(events)
    test_cart_aware_recommender(events)
    test_unified_recommender(events)

if __name__ == "__main__":
    main()
