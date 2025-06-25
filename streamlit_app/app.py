import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import pickle
from recommender.model import UnifiedRecommender
from recommender.embeddings import recommend_with_embeddings

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/events_cleaned.csv")

@st.cache_resource
def load_model(events):
    model = UnifiedRecommender(events)
    model.fit()
    return model

@st.cache_resource
def load_embeddings():
    with open("data/processed/user_embeddings.pkl", "rb") as f:
        user_embeddings = pickle.load(f)
    with open("data/processed/product_embeddings.pkl", "rb") as f:
        product_embeddings = pickle.load(f)
    return user_embeddings, product_embeddings

def main():
    st.title("🛒 E-commerce Recommender Demo")

    tab1, tab2 = st.tabs(["Cart-Based Recommender", "Embedding-Based Recommender"])

    with tab1:
        st.markdown("### Cart-Based Recommender")
        events = load_data()
        model = load_model(events)

        cart_input = st.text_input("Enter cart item IDs (comma-separated):", value="").strip()
        if cart_input:
            cart_items = [int(i.strip()) for i in cart_input.split(",") if i.strip().isdigit()]
            if cart_items:
                recommendations = model.recommend(cart_items=cart_items, top_n=5)
                st.markdown("#### Recommendations:")
                for idx, rec in enumerate(recommendations, 1):
                    st.write(f"{idx}. Item ID: {rec}")
            else:
                st.warning("Invalid item IDs.")
        else:
            st.info("Enter item IDs to get recommendations.")

    with tab2:
        st.markdown("### Embedding-Based Recommender")
        user_id_input = st.text_input("Enter User ID:", value="").strip()
        if user_id_input and user_id_input.isdigit():
            user_id = int(user_id_input)
            user_embeddings, product_embeddings = load_embeddings()

            if user_id in user_embeddings:
                recs = recommend_with_embeddings(user_id, user_embeddings, product_embeddings, top_n=5)
                st.markdown("#### Recommendations:")
                for idx, rec in enumerate(recs, 1):
                    st.write(f"{idx}. Item ID: {rec}")
            else:
                st.warning("User ID not found in embeddings.")
        else:
            st.info("Enter a numeric User ID to see recommendations.")

if __name__ == "__main__":
    main()
