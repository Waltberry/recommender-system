import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

def build_embeddings(embedding_dim=64):
    print("Simulating training data & embeddings...")

    events = pd.read_csv("data/processed/events_cleaned.csv")

    # Interaction scores
    interaction_map = {"view": 1, "addtocart": 3, "transaction": 5}
    events["score"] = events["event"].map(interaction_map)

    users = events["visitorid"].unique()
    items = events["itemid"].unique()

    np.random.seed(42)

    user_embeddings = {
        user: np.random.rand(embedding_dim).astype(np.float32)
        for user in users
    }

    product_embeddings = {
        item: np.random.rand(embedding_dim).astype(np.float32)
        for item in items
    }

    with open("data/processed/user_embeddings.pkl", "wb") as f:
        pickle.dump(user_embeddings, f)

    with open("data/processed/product_embeddings.pkl", "wb") as f:
        pickle.dump(product_embeddings, f)

    print("✅ Saved simulated user & product embeddings.")

if __name__ == "__main__":
    build_embeddings()
