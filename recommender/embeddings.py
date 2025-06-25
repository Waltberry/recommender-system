import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend_with_embeddings(user_id, user_embeddings, product_embeddings, top_n=5):
    if user_id not in user_embeddings:
        return []

    user_vec = user_embeddings[user_id].reshape(1, -1)
    product_ids = list(product_embeddings.keys())
    product_vecs = np.array([product_embeddings[pid] for pid in product_ids])

    sims = cosine_similarity(user_vec, product_vecs).flatten()
    top_indices = sims.argsort()[::-1][:top_n]

    return [product_ids[i] for i in top_indices]

# import faiss

# pip install faiss-cpu


# def recommend_with_faiss(user_id, user_embeddings, product_embeddings, top_n=5):
#     if user_id not in user_embeddings:
#         return []

#     product_ids = list(product_embeddings.keys())
#     vecs = np.stack([product_embeddings[pid] for pid in product_ids]).astype('float32')

#     index = faiss.IndexFlatL2(vecs.shape[1])
#     index.add(vecs)

#     user_vec = np.array([user_embeddings[user_id]]).astype('float32')
#     D, I = index.search(user_vec, top_n)

#     return [product_ids[i] for i in I[0]]
