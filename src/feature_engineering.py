'''import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from transformers import AutoTokenizer

class FeatureEngineering:
    def __init__(self, data_dir="data", models_dir="models"):
        """Initialize the FeatureEngineering class."""
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.processed_train_path = os.path.join(data_dir, "processed", "train_processed.csv")
        self.processed_val_path = os.path.join(data_dir, "processed", "val_processed.csv")
        self.processed_test_path = os.path.join(data_dir, "processed", "test_processed.csv")
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
    def load_processed_data(self):
        """Load preprocessed data."""
        print("Loading preprocessed data...")
        self.train_df = pd.read_csv(self.processed_train_path)
        self.val_df = pd.read_csv(self.processed_val_path)
        self.test_df = pd.read_csv(self.processed_test_path)
        
        print(f"Preprocessed train data shape: {self.train_df.shape}")
        print(f"Preprocessed validation data shape: {self.val_df.shape}")
        print(f"Preprocessed test data shape: {self.test_df.shape}")
        
        return self.train_df, self.val_df, self.test_df
    
    def create_tfidf_features(self, max_features=5000):
        """Create TF-IDF features from cleaned text."""
        print(f"\nCreating TF-IDF features with max_features={max_features}...")
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        
        # Fit and transform training data
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(self.train_df['cleaned_text'])
        
        # Transform validation data
        X_val_tfidf = self.tfidf_vectorizer.transform(self.val_df['cleaned_text'])
        
        # Transform test data
        X_test_tfidf = self.tfidf_vectorizer.transform(self.test_df['cleaned_text'])
        
        # Save vectorizer
        with open(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        print(f"TF-IDF vectorizer saved to {os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')}")
        print(f"Train TF-IDF shape: {X_train_tfidf.shape}")
        print(f"Validation TF-IDF shape: {X_val_tfidf.shape}")
        print(f"Test TF-IDF shape: {X_test_tfidf.shape}")
        
        return X_train_tfidf, X_val_tfidf, X_test_tfidf
    
    def encode_labels(self):
        """Encode labels into numerical format."""
        print("\nEncoding labels...")
        
        # Fit label encoder on training data
        y_train = self.label_encoder.fit_transform(self.train_df['Labels'])
        
        # Transform validation labels if they exist
        y_val = None
        if 'Labels' in self.val_df.columns:
            y_val = self.label_encoder.transform(self.val_df['Labels'])
        
        # Save label encoder
        with open(os.path.join(self.models_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"Label encoder saved to {os.path.join(self.models_dir, 'label_encoder.pkl')}")
        print(f"Encoded labels: {self.label_encoder.classes_}")
        
        return y_train, y_val
    
    def prepare_bert_inputs(self, model_name="bert-base-uncased", max_length=128):
        """Prepare inputs for BERT model."""
        print(f"\nPreparing BERT inputs using {model_name}...")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Tokenize training data
        train_encodings = tokenizer(
            self.train_df['Interview Text'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Tokenize validation data
        val_encodings = tokenizer(
            self.val_df['Interview Text'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Tokenize test data
        test_encodings = tokenizer(
            self.test_df['Interview Text'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.models_dir, 'bert_tokenizer'))
        
        print(f"BERT tokenizer saved to {os.path.join(self.models_dir, 'bert_tokenizer')}")
        
        return train_encodings, val_encodings, test_encodings


if __name__ == "__main__":
    # Initialize feature engineering
    feature_eng = FeatureEngineering()
    
    # Load processed data
    train_df, val_df, test_df = feature_eng.load_processed_data()
    
    # Create TF-IDF features
    X_train_tfidf, X_val_tfidf, X_test_tfidf = feature_eng.create_tfidf_features()
    
    # Encode labels
    y_train, y_val = feature_eng.encode_labels()
    
    # Prepare BERT inputs
    train_encodings, val_encodings, test_encodings = feature_eng.prepare_bert_inputs()
    
    
    
    
    
    
    
    
    
    
    
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import torch
from transformers import AutoTokenizer

class FeatureEngineering:
    def __init__(self, data_dir="data", models_dir="models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.processed_train_path = os.path.join(data_dir, "processed", "train_processed.csv")
        self.processed_val_path = os.path.join(data_dir, "processed", "val_processed.csv")
        self.processed_test_path = os.path.join(data_dir, "processed", "test_processed.csv")

        os.makedirs(models_dir, exist_ok=True)
        self.label_encoder = LabelEncoder()

    def load_processed_data(self):
        print("Loading preprocessed data...")
        self.train_df = pd.read_csv(self.processed_train_path)
        self.val_df = pd.read_csv(self.processed_val_path)
        self.test_df = pd.read_csv(self.processed_test_path)

        # Drop rows with missing text if any
        self.train_df.dropna(subset=['Interview Text'], inplace=True)
        self.val_df.dropna(subset=['Interview Text'], inplace=True)
        self.test_df.dropna(subset=['Interview Text'], inplace=True)

        print(f"Train shape: {self.train_df.shape}, Val shape: {self.val_df.shape}, Test shape: {self.test_df.shape}")
        return self.train_df, self.val_df, self.test_df

    def create_tfidf_features(self, max_features=5000):
        print(f"\nCreating TF-IDF features with max_features={max_features}...")
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        X_train = self.tfidf_vectorizer.fit_transform(self.train_df['cleaned_text'])
        X_val = self.tfidf_vectorizer.transform(self.val_df['cleaned_text'])
        X_test = self.tfidf_vectorizer.transform(self.test_df['cleaned_text'])

        with open(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)

        print("TF-IDF vectorizer saved.")
        return X_train, X_val, X_test

    def encode_labels(self):
        print("\nEncoding labels...")
        y_train = self.label_encoder.fit_transform(self.train_df['Labels'])

        y_val = None
        if 'Labels' in self.val_df.columns and self.val_df['Labels'].notnull().all():
            y_val = self.label_encoder.transform(self.val_df['Labels'])
        else:
            print("Warning: Validation labels are missing or incomplete.")

        with open(os.path.join(self.models_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)

        print("Label encoder saved.")
        return y_train, y_val

    def prepare_bert_inputs(self, model_name="bert-base-uncased", max_length=128):
        print(f"\nPreparing BERT inputs using {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_encodings = tokenizer(
            self.train_df['Interview Text'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        val_encodings = tokenizer(
            self.val_df['Interview Text'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        test_encodings = tokenizer(
            self.test_df['Interview Text'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        tokenizer.save_pretrained(os.path.join(self.models_dir, 'bert_tokenizer'))

        # Optional: save tensors
        # torch.save(train_encodings, os.path.join(self.models_dir, 'train_bert_inputs.pt'))

        print("BERT tokenizer and encodings prepared.")
        return train_encodings, val_encodings, test_encodings

if __name__ == "__main__":
    fe = FeatureEngineering()
    train_df, val_df, test_df = fe.load_processed_data()
    X_train_tfidf, X_val_tfidf, X_test_tfidf = fe.create_tfidf_features()
    y_train, y_val = fe.encode_labels()
    train_enc, val_enc, test_enc = fe.prepare_bert_inputs()
    
    
    
    '''
    
#src/feature_engineering.py  
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from transformers import AutoTokenizer


class FeatureEngineering:
    def __init__(self, data_dir="data", models_dir="models"):
        """Initialize the FeatureEngineering class."""
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.processed_train_path = os.path.join(data_dir, "processed", "train_processed.csv")
        self.processed_val_path = os.path.join(data_dir, "processed", "val_processed.csv")
        self.processed_test_path = os.path.join(data_dir, "processed", "test_processed.csv")

        os.makedirs(models_dir, exist_ok=True)
        self.label_encoder = LabelEncoder()

    def load_processed_data(self):
        """Load preprocessed data."""
        print("Loading preprocessed data...")
        self.train_df = pd.read_csv(self.processed_train_path)
        self.val_df = pd.read_csv(self.processed_val_path)
        self.test_df = pd.read_csv(self.processed_test_path)

        print(f"Preprocessed train data shape: {self.train_df.shape}")
        print(f"Preprocessed validation data shape: {self.val_df.shape}")
        print(f"Preprocessed test data shape: {self.test_df.shape}")

        return self.train_df, self.val_df, self.test_df

    def create_tfidf_features(self, max_features=10000):
        """Create TF-IDF features from cleaned text."""
        print(f"\nCreating TF-IDF features with max_features={max_features}...")

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=2
        )

        X_train_tfidf = self.tfidf_vectorizer.fit_transform(self.train_df['cleaned_text'])
        X_val_tfidf = self.tfidf_vectorizer.transform(self.val_df['cleaned_text'])
        X_test_tfidf = self.tfidf_vectorizer.transform(self.test_df['cleaned_text'])

        with open(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)

        print(f"TF-IDF vectorizer saved to {os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')}")
        print(f"Train TF-IDF shape: {X_train_tfidf.shape}")
        print(f"Validation TF-IDF shape: {X_val_tfidf.shape}")
        print(f"Test TF-IDF shape: {X_test_tfidf.shape}")

        self.X_train_tfidf = X_train_tfidf
        self.X_val_tfidf = X_val_tfidf
        self.X_test_tfidf = X_test_tfidf

        return X_train_tfidf, X_val_tfidf, X_test_tfidf

    def encode_labels(self):
        """Encode labels into numerical format."""
        print("\nEncoding labels...")

        y_train = self.label_encoder.fit_transform(self.train_df['Labels'])

        y_val = None
        if 'Labels' in self.val_df.columns:
            y_val = self.label_encoder.transform(self.val_df['Labels'])

        with open(os.path.join(self.models_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)

        print(f"Label encoder saved to {os.path.join(self.models_dir, 'label_encoder.pkl')}")
        print(f"Encoded labels: {self.label_encoder.classes_}")

        # Store as instance attributes
        self.y_train = y_train
        self.y_val = y_val

        return y_train, y_val

    def prepare_bert_inputs(self, model_name="bert-base-uncased", max_length=256):
        """Prepare inputs for BERT or any HuggingFace model."""
        print(f"\nPreparing transformer inputs using {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_encodings = tokenizer(
            self.train_df['Interview Text'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        val_encodings = tokenizer(
            self.val_df['Interview Text'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        test_encodings = tokenizer(
            self.test_df['Interview Text'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        tokenizer.save_pretrained(os.path.join(self.models_dir, 'bert_tokenizer'))

        print(f"Tokenizer saved to {os.path.join(self.models_dir, 'bert_tokenizer')}")

        # Store as instance attributes
        self.train_encodings = train_encodings
        self.val_encodings = val_encodings
        self.test_encodings = test_encodings

        return train_encodings, val_encodings, test_encodings


if __name__ == "__main__":
    # Example run with custom model
    fe = FeatureEngineering()
    fe.load_processed_data()
    fe.create_tfidf_features()
    fe.encode_labels()
    fe.prepare_bert_inputs(model_name="roberta-base")