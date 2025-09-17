'''import os
import re
import logging
from collections import Counter

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources once
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class DataPreprocessor:
    def __init__(self, data_dir="data", processed_subdir="processed", figures_dir="results/figures"):
        """
        Initialize the preprocessor with paths and tools.
        """
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, processed_subdir)
        self.figures_dir = figures_dir
        
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        self.train_path = os.path.join(data_dir, "train.csv")
        self.val_path = os.path.join(data_dir, "val.csv")
        self.test_path = os.path.join(data_dir, "test.csv")
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        self.train_df = None
        self.val_df = None
        self.test_df = None
    
    def load_data(self):
        """
        Load CSV files into pandas DataFrames.
        """
        logging.info("Loading datasets...")
        self.train_df = pd.read_csv(self.train_path)
        self.val_df = pd.read_csv(self.val_path)
        self.test_df = pd.read_csv(self.test_path)
        
        logging.info(f"Train: {self.train_df.shape}, Val: {self.val_df.shape}, Test: {self.test_df.shape}")
        return self.train_df, self.val_df, self.test_df

    def explore_data(self, df, is_training=True):
        """
        Print basic dataset info, missing values, label distribution, etc.
        """
        logging.info(f"Columns: {df.columns.tolist()}")
        logging.info(f"Sample rows:\n{df.head(2)}")
        
        missing = df.isnull().sum()
        logging.info(f"Missing values:\n{missing}")
        
        if is_training and 'Labels' in df.columns:
            label_counts = df['Labels'].value_counts()
            logging.info(f"Label Distribution:\n{label_counts}")
            df['text_length'] = df['Interview Text'].apply(lambda x: len(str(x)))
            logging.info(f"Text length stats:\n{df['text_length'].describe()}")

    def clean_text(self, text):
        """
        Clean, tokenize, remove stopwords, and lemmatize text.
        """
        if pd.isna(text):
            return ""
        text = text.lower().strip()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return " ".join(tokens)

    def preprocess_data(self):
        """
        Apply text cleaning to all datasets.
        """
        logging.info("Preprocessing train/val/test datasets...")
        self.train_df['cleaned_text'] = self.train_df['Interview Text'].apply(self.clean_text)
        self.val_df['cleaned_text'] = self.val_df['Interview Text'].apply(self.clean_text)
        self.test_df['cleaned_text'] = self.test_df['Interview Text'].apply(self.clean_text)

        self.train_df.to_csv(os.path.join(self.processed_dir, "train_processed.csv"), index=False)
        self.val_df.to_csv(os.path.join(self.processed_dir, "val_processed.csv"), index=False)
        self.test_df.to_csv(os.path.join(self.processed_dir, "test_processed.csv"), index=False)
        logging.info("Saved cleaned data to processed/ directory.")
        return self.train_df, self.val_df, self.test_df

    def visualize_class_distribution(self):
        """
        Bar plot of training label distribution.
        """
        logging.info("Generating class distribution plot...")
        plt.figure(figsize=(12, 8))
        sns.countplot(y=self.train_df['Labels'], order=self.train_df['Labels'].value_counts().index)
        plt.title("Distribution of Interview Categories")
        plt.xlabel("Count")
        plt.ylabel("Category")
        plt.tight_layout()
        fig_path = os.path.join(self.figures_dir, "class_distribution.png")
        plt.savefig(fig_path)
        plt.close()
        logging.info(f"Saved class distribution to {fig_path}")

    def create_word_clouds(self):
        """
        Generate and save word clouds for each class.
        """
        logging.info("Generating word clouds...")
        for label in self.train_df['Labels'].unique():
            subset = self.train_df[self.train_df['Labels'] == label]
            combined_text = " ".join(subset['cleaned_text'])
            wc = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(combined_text)
            plt.figure(figsize=(10, 6))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Word Cloud - Label: {label}")
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, f"wordcloud_label_{label}.png"))
            plt.close()
        logging.info(f"Saved word clouds to {self.figures_dir}")

    def analyze_token_frequency(self):
        """
        Plot token frequency per label.
        """
        logging.info("Analyzing token frequencies...")
        for label in self.train_df['Labels'].unique():
            subset = self.train_df[self.train_df['Labels'] == label]
            tokens = word_tokenize(" ".join(subset['cleaned_text']))
            freq = Counter(tokens).most_common(20)
            words, counts = zip(*freq)
            plt.figure(figsize=(12, 8))
            sns.barplot(x=list(counts), y=list(words))
            plt.title(f"Top 20 Tokens for Label: {label}")
            plt.xlabel("Frequency")
            plt.ylabel("Words")
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, f"token_freq_label_{label}.png"))
            plt.close()
        logging.info("Token frequency plots saved.")

# Example usage
if __name__ == "__main__":
    dp = DataPreprocessor()
    train_df, val_df, test_df = dp.load_data()
    dp.explore_data(train_df)
    train_df, val_df, test_df = dp.preprocess_data()
    dp.visualize_class_distribution()
    dp.create_word_clouds()
    dp.analyze_token_frequency()
'''

# src/data_preprocessing.py
import os
import re
import logging
from collections import Counter

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import contractions

# Download NLTK resources once
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

LABEL_MAP = {
    1: "Game Strategy",
    2: "Player Performance",
    3: "Injury Updates",
    4: "Post-Game Analysis",
    5: "Team Morale",
    6: "Upcoming Matches",
    7: "Off-Game Matters",
    8: "Controversies"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    def __init__(self, data_dir="data", results_dir="results", processed_subdir="processed"):
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, processed_subdir)
        self.figures_dir = os.path.join(results_dir, "figures")

        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        self.train_path = os.path.join(data_dir, "train.csv")
        self.val_path = os.path.join(data_dir, "val.csv")
        self.test_path = os.path.join(data_dir, "test.csv")

        self.stop_words = set(stopwords.words('english')) - {'what', 'why', 'how', 'who', 'when', 'where'}
        self.lemmatizer = WordNetLemmatizer()

        self.train_df = None
        self.val_df = None
        self.test_df = None

    def load_data(self):
        logging.info("Loading datasets...")
        self.train_df = pd.read_csv(self.train_path)
        self.val_df = pd.read_csv(self.val_path)
        self.test_df = pd.read_csv(self.test_path)

        logging.info(f"Train: {self.train_df.shape}, Val: {self.val_df.shape}, Test: {self.test_df.shape}")
        return self.train_df, self.val_df, self.test_df

    def explore_data(self, df, is_training=True):
        logging.info(f"Columns: {df.columns.tolist()}")
        logging.info(f"Sample rows:\n{df.head(2)}")
        missing = df.isnull().sum()
        logging.info(f"Missing values:\n{missing}")

        if is_training and 'Labels' in df.columns:
            label_counts = df['Labels'].value_counts()
            logging.info(f"Label Distribution:\n{label_counts}")
            df['text_length'] = df['Interview Text'].apply(lambda x: len(str(x)))
            logging.info(f"Text length stats:\n{df['text_length'].describe()}")

    def clean_text(self, text):
        if pd.isna(text):
            return ""

        text = contractions.fix(text)
        text = text.lower()
        text = re.sub(r"interviewer:|participant:|patient:|doctor:", "", text, flags=re.IGNORECASE)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        tokens = word_tokenize(text)
        cleaned_tokens = []
        for word in tokens:
            if word.isalpha() and word not in self.stop_words:
                lemma = self.lemmatizer.lemmatize(word)
                if len(lemma) > 2:
                    cleaned_tokens.append(lemma)

        return " ".join(cleaned_tokens)

    def clean_text_single(self, text):
        return self.clean_text(text)

    def preprocess_data(self):
        logging.info("Preprocessing train/val/test datasets...")

        for df_name, df in zip(['train_df', 'val_df', 'test_df'], [self.train_df, self.val_df, self.test_df]):
            df['cleaned_text'] = df['Interview Text'].apply(self.clean_text)
            df['tokens'] = df['cleaned_text'].apply(lambda x: x.split())

        assert not self.train_df['cleaned_text'].isnull().any(), "Null values found in train cleaned text!"

        self.train_df.to_csv(os.path.join(self.processed_dir, "train_processed.csv"), index=False)
        self.val_df.to_csv(os.path.join(self.processed_dir, "val_processed.csv"), index=False)
        self.test_df.to_csv(os.path.join(self.processed_dir, "test_processed.csv"), index=False)

        logging.info("Saved cleaned data to processed/ directory.")
        return self.train_df, self.val_df, self.test_df

    def visualize_class_distribution(self):
        logging.info("Generating class distribution plot...")
        plt.figure(figsize=(12, 8))
        sns.countplot(y=self.train_df['Labels'], order=self.train_df['Labels'].value_counts().index)
        plt.title("Distribution of Interview Categories")
        plt.xlabel("Count")
        plt.ylabel("Category")
        plt.tight_layout()
        fig_path = os.path.join(self.figures_dir, "class_distribution.png")
        plt.savefig(fig_path)
        plt.close()
        logging.info(f"Saved class distribution to {fig_path}")

    def create_word_clouds(self):
        logging.info("Generating word clouds...")
        for label in self.train_df['Labels'].unique():
            subset = self.train_df[self.train_df['Labels'] == label]
            combined_text = " ".join(subset['cleaned_text'])
            wc = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(combined_text)
            plt.figure(figsize=(10, 6))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Word Cloud - {LABEL_MAP.get(label, label)}")
            plt.tight_layout()
            path = os.path.join(self.figures_dir, f"wordcloud_label_{label}.png")
            plt.savefig(path)
            plt.close()
        logging.info("Saved word clouds.")

    def analyze_token_frequency(self):
        logging.info("Analyzing token frequencies...")
        for label in self.train_df['Labels'].unique():
            subset = self.train_df[self.train_df['Labels'] == label]
            tokens = [token for sublist in subset['tokens'] for token in sublist]
            freq = Counter(tokens).most_common(20)
            words, counts = zip(*freq)
            plt.figure(figsize=(12, 8))
            sns.barplot(x=list(counts), y=list(words))
            plt.title(f"Top 20 Tokens - {LABEL_MAP.get(label, label)}")
            plt.xlabel("Frequency")
            plt.ylabel("Words")
            plt.tight_layout()
            path = os.path.join(self.figures_dir, f"token_freq_label_{label}.png")
            plt.savefig(path)
            plt.close()
        logging.info("Token frequency plots saved.")

if __name__ == "__main__":
    dp = DataPreprocessor()
    train_df, val_df, test_df = dp.load_data()
    dp.explore_data(train_df)
    train_df, val_df, test_df = dp.preprocess_data()
    dp.visualize_class_distribution()
    dp.create_word_clouds()
    dp.analyze_token_frequency()
