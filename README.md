# 🛒 E-Commerce Recommender System

A modular, production-ready product recommendation engine built using **Python** and **Streamlit**. This system is designed to suggest relevant items based on real-time and historical user behavior—such as browsing history, purchase activity, and cart contents.

Using real-world data from **RetailRocket**, the system demonstrates foundational recommendation approaches including **popularity-based**, **item-to-item similarity**, and a **cart-aware hybrid method**. It simulates both batch and streaming pipelines to reflect a scalable, real-world architecture for modern e-commerce platforms.

---

## 🚀 Features

* ✅ Popularity-based recommender
* ✅ Item-to-item similarity engine
* ✅ Cart-aware hybrid recommender
* ✅ Streamlit web interface for interactive demos
* ✅ Modular, extensible architecture
* ✅ Real-world user-event dataset for realistic simulation

---

## 📊 Dataset

This project uses the [RetailRocket E-Commerce Dataset](https://www.kaggle.com/retailrocket/ecommerce-dataset), which contains detailed records of user interactions in an online retail environment.

### 📥 Setup Instructions:

1. Download the dataset from [Kaggle](https://www.kaggle.com/retailrocket/ecommerce-dataset)
2. Extract the contents into the `data/raw/` directory:

```
data/
└── raw/
    ├── events.csv
    ├── item_properties_part1.csv
    └── item_properties_part2.csv
```

⚠️ Note: These files are large (100MB+), so they are **not** tracked via Git.

---

## 🧠 Recommendation Techniques

### 1. Popularity-Based Recommender

Recommends the most frequently purchased products across all users—ideal for new users without history (cold-start problem).

### 2. Item-to-Item Similarity Recommender

Generates suggestions based on co-occurrence patterns in user sessions. Items frequently purchased/viewed together are deemed similar.

### 3. Cart-Aware Hybrid Recommender

A hybrid method that incorporates:

* Items in the user’s cart
* Item similarity-based recommendations
* Popular fallback items when user history is sparse

---

## 🖥️ Running the Project

### 🔧 Setup Environment

```bash
git clone https://github.com/yourusername/recommender-system.git
cd recommender-system
python -m venv env
source env/bin/activate  # or .\env\Scripts\activate on Windows
pip install -r requirements.txt
```

### 🧪 Run the Model via Script

```bash
python main.py
```

### 🌐 Launch the Streamlit Demo

```bash
streamlit run streamlit_app/app.py
```

---

## 📁 Project Structure

```
recommender-system/
├── batch_pipeline/              # For model training data prep
│   └── build_training_data.py
│
├── streaming_pipeline/          # For real-time recs simulation
│   └── simulate_stream.py
│
├── recommender/
│   └── model.py                 # Popularity, item-item, unified
│   └── embeddings.py            # (new) vector-based model utils
│
├── vector_db/                   # (optional) FAISS or vector search
│   └── fake_vector_store.py     # Dummy vector DB for local use
│
├── data/
│   └── raw/
│   └── processed/
│       └── training_data.csv    # From batch pipeline
│       └── product_embeddings.pkl
│       └── user_embeddings.pkl
│
├── streamlit_app/
│   └── app.py
│
├── notebooks/             # Jupyter notebooks for EDA and prototyping
├── main.py                # Script for testing the models
├── requirements.txt
└── README.md
```

---

## 🧩 Batch & Streaming Pipelines

This project simulates a robust, end-to-end architecture for a scalable recommendation system.

### 🧱 Batch Pipeline (`batch_pipeline/`)

Prepares embeddings from historical interaction data:

* Extracts user behavior (views, cart additions, purchases)
* Simulates 64-dimensional user/product embeddings
* Saves:

  * `user_embeddings.pkl`
  * `product_embeddings.pkl`

```bash
python batch_pipeline/build_training_data.py
```

### ⚡ Streaming Pipeline (`streaming_pipeline/`)

Simulates real-time recommendation delivery based on active user sessions:

* Retrieves precomputed user embeddings
* Calculates top-N recommendations using cosine similarity

```bash
python streaming_pipeline/simulate_stream.py
```

---

## 📚 Skills & Concepts Demonstrated

* Designing content-based, popularity-based, and hybrid recommender systems
* Preprocessing real-world user-event data
* Building modular Python components
* Developing interactive dashboards with Streamlit
* Simulating batch and streaming ML pipelines
* Vector-based recommendation logic using embeddings

---

## 📌 Future Improvements

* Incorporate collaborative filtering (e.g., matrix factorization, ALS)
* Improve scalability using sparse matrices or approximate nearest neighbors (FAISS, Annoy, etc.)
* Add session-based recommenders (RNNs or Transformers)
* Use product metadata to build a content-based recommender

---

## 🙋‍♂️ About the Author

**Onyero Walter Ofuzim**
*Back-End Data Engineer | Data Professional | Software Developer*

* 🔗 [LinkedIn](https://www.linkedin.com/in/onyero-walter-ofuzim-189301107/)
* 🖥 [GitHub](https://github.com/Waltberry)

---

## 📝 License

This project is released under the [MIT License](LICENSE).

---
