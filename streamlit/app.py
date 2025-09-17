#src: streamlit/app.py
import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import pickle
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 1. Configure App ---
st.set_page_config(page_title="Chasuite Sports Interview Analyzer", layout="wide")
st.title("🏆 Sports Interview Analysis Dashboard")

# --- 2. Load Models ---
@st.cache_resource
def load_models():
    # Load BERT
    bert_classifier = pipeline(
        "text-classification", 
        model="./models/bert_model",
        tokenizer="distilbert-base-uncased"
    )
    
    # Load TF-IDF + Logistic Regression
    with open("./models/tfidf_model.pkl", "rb") as f:
        tfidf, lr_model = pickle.load(f)
    
    return bert_classifier, tfidf, lr_model

bert_model, tfidf, lr_model = load_models()

# --- 3. Classification Tab ---
tab1, tab2, tab3 = st.tabs(["Classify", "Q&A Generator", "Data Explorer"])

with tab1:
    st.header("Interview Transcript Classifier")
    user_input = st.text_area("Paste interview transcript here:", height=200)
    
    if st.button("Classify"):
        # Preprocess input
        cleaned_text = clean_text(user_input)  # Your preprocessing function
        
        # Get predictions
        bert_pred = bert_model(cleaned_text)[0]["label"]
        tfidf_vec = tfidf.transform([cleaned_text])
        lr_pred = lr_model.predict(tfidf_vec)[0]
        
        st.success(f"BERT Prediction: {bert_pred}")
        st.info(f"TF-IDF + Logistic Regression Prediction: {lr_pred}")

# --- 4. Q&A Generator (Section C) ---
with tab2:
    st.header("AI Interview Response Generator")
    col1, col2 = st.columns(2)
    
    with col1:
        category = st.selectbox(
            "Select interview type:",
            options=["Game Strategy", "Player Performance", "Injury Updates"]
        )
    
    with col2:
        question = st.text_input("Enter your question:")
    
    if st.button("Generate Response"):
        # Simulated LLM response (replace with your actual text generation)
        response = f"**{category} Response**: This is a simulated answer to '{question}'. In real implementation, use GPT-3 or fine-tuned LLM."
        st.markdown(response)

# --- 5. Data Visualization (Section D) ---
with tab3:
    st.header("Interview Topic Clustering")
    
    # Load sample processed data
    sample_data = pd.read_csv("./data/processed/train_clean.csv").sample(100)
    
    # Reduce dimensions with UMAP
    umap = UMAP(n_components=2, random_state=42)
    embeddings = umap.fit_transform(tfidf.transform(sample_data["cleaned_text"]))
    
    # Plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=sample_data["Labels"], alpha=0.5)
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    st.pyplot(fig)