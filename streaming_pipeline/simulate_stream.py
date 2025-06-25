import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pickle
from recommender.embeddings import recommend_with_embeddings

def simulate_stream():
    # Load product/user embeddings
    with open("data/processed/user_embeddings.pkl", "rb") as f:
        user_embeddings = pickle.load(f)

    with open("data/processed/product_embeddings.pkl", "rb") as f:
        product_embeddings = pickle.load(f)

    # Simulate live user activity
    events = pd.read_csv("data/processed/events_cleaned.csv")
    test_users = events["visitorid"].unique()[:10]

    for user in test_users:
        print(f"\n🧑 User {user}:")
        if user in user_embeddings:
            recs = recommend_with_embeddings(user, user_embeddings, product_embeddings, top_n=5)
            print("Top Recommendations:", recs)
        else:
            print("No embedding found. Using fallback.")

if __name__ == "__main__":
    simulate_stream()
