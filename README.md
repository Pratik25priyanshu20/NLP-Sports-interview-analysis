# Chasuite Interview Classification & Analysis

This repository contains the complete solution to the Chasuite NLP Competition, which focuses on analyzing professional sports interview transcripts and classifying them into predefined categories using state-of-the-art NLP techniques.

## 🎯 Objective

Given a dataset of interview transcripts across sports contexts (e.g., NFL, NBA, NHL), the goal is to:

- Predict interview categories like "Game Strategy", "Post-Game Analysis", etc.
- Generate plausible interview responses based on category and question
- Visualize clusters of topics within the interviews
- Build a Streamlit dashboard to demo the system

---

## 🧱 Project Structurechasuite_exam/
│
├── data/                   # Raw and processed data
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── processed/
│
├── models/                 # Trained models and tokenizers
│   ├── logistic_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── bert_model/
│
├── results/                # Outputs: predictions, figures, visualizations
│   ├── submission.csv
│   ├── results.json
│   ├── figures/
│   ├── embeddings/
│   └── text_generation/
│
├── streamlit/              # Streamlit app and UI components
│   ├── app.py
│   └── components/
│
├── src/                    # Source code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── text_generation.py
│   └── visualization.py
│
├── main.py                 # End-to-end driver script
├── requirements.txt        # Python dependencies
├── README.md               # You’re here
└── AI_tool_declaration.txt # Declaration of AI tool usage



---

## 🔧 Technologies Used

- **Python 3.9+**
- **Transformers (HuggingFace)** for BERT and GPT-2
- **scikit-learn** for classic ML models
- **NLTK** and **WordCloud** for text preprocessing and visualization
- **UMAP, t-SNE, KMeans** for topic clustering
- **Streamlit** for dashboard
- **Plotly** for interactive embeddings

---

## 🚀 Key Features

- ✅ Logistic Regression with TF-IDF and feature importance
- ✅ Fine-tuned BERT for multi-class classification
- ✅ GPT-2-based response generation
- ✅ UMAP-based topic clustering and visualization
- ✅ Ethical reflection on AI in journalism
- ✅ Streamlit app with full functionality

---

## 📝 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt



Run the pipeline 

python main.py


Launch the StreamlitApp

streamlit run streamlit/app.py