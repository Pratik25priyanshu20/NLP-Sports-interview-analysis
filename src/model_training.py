'''import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset

class ModelTrainer:
    def __init__(self, models_dir="models", results_dir="results"):
        """Initialize the ModelTrainer class."""
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # Create directories if they don't exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    
    def train_logistic_regression(self, X_train, y_train, X_val=None, y_val=None, 
                                  C=1.0, max_iter=1000, class_weight='balanced'):
        """Train a logistic regression model."""
        print(f"\nTraining Logistic Regression model with C={C}, max_iter={max_iter}...")
        
        # Initialize and train model
        self.log_reg = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42
        )
        self.log_reg.fit(X_train, y_train)
        
        # Save model
        with open(os.path.join(self.models_dir, 'logistic_model.pkl'), 'wb') as f:
            pickle.dump(self.log_reg, f)
        
        print(f"Logistic Regression model saved to {os.path.join(self.models_dir, 'logistic_model.pkl')}")
        
        # Evaluate if validation data is provided
        if X_val is not None and y_val is not None:
            y_val_pred = self.log_reg.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted')
            
            print(f"Validation accuracy: {val_accuracy:.4f}")
            print(f"Validation weighted F1 score: {val_f1:.4f}")
            
            # Create confusion matrix
            cm = confusion_matrix(y_val, y_val_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - Logistic Regression')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(self.results_dir, "figures", "logreg_confusion_matrix.png"))
            plt.close()
            
            print(f"Confusion matrix saved to {os.path.join(self.results_dir, 'figures', 'logreg_confusion_matrix.png')}")
            
            # Save evaluation results
            results = {
                'model': 'Logistic Regression',
                'accuracy': val_accuracy,
                'weighted_f1': val_f1
            }
            
            with open(os.path.join(self.results_dir, 'logreg_results.pkl'), 'wb') as f:
                pickle.dump(results, f)
            
            return val_accuracy, val_f1
        
        return None, None
    
    def analyze_logreg_features(self, vectorizer, label_encoder, top_n=10):
        """Analyze feature importance in logistic regression model."""
        print(f"\nAnalyzing top {top_n} features per class for Logistic Regression...")
        
        feature_names = vectorizer.get_feature_names_out()
        class_labels = label_encoder.classes_
        
        # Create DataFrame to store feature importances
        feature_importance_df = pd.DataFrame()
        
        # For each class
        for i, class_label in enumerate(class_labels):
            # Get coefficients for class i
            coef = self.log_reg.coef_[i]
            
            # Sort coefficients
            top_indices = coef.argsort()[-top_n:][::-1]
            top_features = [feature_names[idx] for idx in top_indices]
            top_values = coef[top_indices]
            
            # Create temporary DataFrame
            temp_df = pd.DataFrame({
                'feature': top_features,
                'importance': top_values,
                'class': class_label
            })
            
            # Append to main DataFrame
            feature_importance_df = pd.concat([feature_importance_df, temp_df])
        
        # Save feature importance
        feature_importance_df.to_csv(os.path.join(self.results_dir, 'logreg_feature_importance.csv'), index=False)
        
        # Plot feature importance for each class
        plt.figure(figsize=(15, 12))
        g = sns.catplot(
            data=feature_importance_df,
            kind='bar',
            x='importance',
            y='feature',
            hue='class',
            col='class',
            col_wrap=3,
            height=4,
            aspect=1.5,
            sharex=False,
            sharey=False
        )
        g.fig.suptitle('Top Features by Class - Logistic Regression', fontsize=16)
        g.fig.subplots_adjust(top=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "figures", "logreg_feature_importance.png"))
        plt.close()
        
        print(f"Feature importance analysis saved to {os.path.join(self.results_dir, 'logreg_feature_importance.csv')}")
        print(f"Feature importance plot saved to {os.path.join(self.results_dir, 'figures', 'logreg_feature_importance.png')}")
    
    def train_bert_model(self, train_encodings, y_train, val_encodings=None, y_val=None,
                         model_name="bert-base-uncased", num_labels=8, epochs=3,
                         batch_size=16, learning_rate=5e-5):
        """Train a BERT model for sequence classification."""
        print(f"\nTraining BERT model ({model_name})...")
        
        # Convert encodings to Dataset format
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'].numpy(),
            'attention_mask': train_encodings['attention_mask'].numpy(),
            'labels': y_train
        })
        
        val_dataset = None
        if val_encodings is not None and y_val is not None:
            val_dataset = Dataset.from_dict({
                'input_ids': val_encodings['input_ids'].numpy(),
                'attention_mask': val_encodings['attention_mask'].numpy(),
                'labels': y_val
            })
        
        # Initialize model
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(self.models_dir, 'bert_checkpoints'),
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="f1" if val_dataset else None,
            push_to_hub=False,
        )
        
        # Define compute_metrics function for evaluation
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            f1 = f1_score(labels, preds, average='weighted')
            acc = accuracy_score(labels, preds)
            return {
                'accuracy': acc,
                'f1': f1
            }
        
        # Initialize trainer
        trainer = Trainer(
            model=self.bert_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics if val_dataset else None
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model(os.path.join(self.models_dir, 'bert_model'))
        print(f"BERT model saved to {os.path.join(self.models_dir, 'bert_model')}")
        
        # Evaluate on validation set
        if val_dataset:
            eval_results = trainer.evaluate()
            print(f"Validation results: {eval_results}")
            
            # Save evaluation results
            results = {
                'model': f'BERT ({model_name})',
                'accuracy': eval_results['eval_accuracy'],
                'weighted_f1': eval_results['eval_f1']
            }
            
            with open(os.path.join(self.results_dir, 'bert_results.pkl'), 'wb') as f:
                pickle.dump(results, f)
            
            # Generate predictions for confusion matrix
            val_preds = trainer.predict(val_dataset)
            y_val_pred = np.argmax(val_preds.predictions, axis=1)
            
            # Create confusion matrix
            cm = confusion_matrix(y_val, y_val_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - BERT')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(self.results_dir, "figures", "bert_confusion_matrix.png"))
            plt.close()
            
            print(f"Confusion matrix saved to {os.path.join(self.results_dir, 'figures', 'bert_confusion_matrix.png')}")
            
            return eval_results['eval_accuracy'], eval_results['eval_f1']
        
        return None, None


if __name__ == "__main__":
    # You will typically call this after feature engineering
    # and have the necessary variables like X_train_tfidf, y_train, etc.
    pass
    
    
    
    
    
import os
import pickle
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
)
from datasets import Dataset

class ModelTrainer:
    def __init__(self, models_dir="models", results_dir="results"):
        self.models_dir = models_dir
        self.results_dir = results_dir

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    def train_logistic_regression(self, X_train, y_train, X_val=None, y_val=None, 
                                  C=1.0, max_iter=1000, class_weight='balanced'):
        print(f"\nTraining Logistic Regression model with C={C}, max_iter={max_iter}...")
        self.log_reg = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42
        )
        self.log_reg.fit(X_train, y_train)

        with open(os.path.join(self.models_dir, 'logistic_model.pkl'), 'wb') as f:
            pickle.dump(self.log_reg, f)

        print("Logistic Regression model saved.")

        if X_val is not None and y_val is not None:
            y_val_pred = self.log_reg.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted')

            print(f"Validation accuracy: {val_accuracy:.4f}")
            print(f"Validation weighted F1 score: {val_f1:.4f}")

            cm = confusion_matrix(y_val, y_val_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - Logistic Regression')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(self.results_dir, "figures", "logreg_confusion_matrix.png"))
            plt.close()

            results = {
                'model': 'Logistic Regression',
                'accuracy': val_accuracy,
                'weighted_f1': val_f1
            }
            with open(os.path.join(self.results_dir, 'logreg_results.pkl'), 'wb') as f:
                pickle.dump(results, f)

            # Save final required results.json
            with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
                json.dump({
                    "f1_score": round(val_f1, 5),
                    "accuracy": round(val_accuracy, 5)
                }, f)

            return val_accuracy, val_f1

        print("Validation data not provided — skipping evaluation.")
        return None, None

    def analyze_logreg_features(self, vectorizer, label_encoder, top_n=10):
        print(f"\nAnalyzing top {top_n} features per class...")
        feature_names = vectorizer.get_feature_names_out()
        class_labels = label_encoder.classes_

        feature_importance_df = pd.DataFrame()

        for i, class_label in enumerate(class_labels):
            coef = self.log_reg.coef_[i]
            top_indices = coef.argsort()[-top_n:][::-1]
            top_features = [feature_names[idx] for idx in top_indices]
            top_values = coef[top_indices]
            temp_df = pd.DataFrame({
                'feature': top_features,
                'importance': top_values,
                'class': class_label
            })
            feature_importance_df = pd.concat([feature_importance_df, temp_df])

        feature_importance_df.to_csv(os.path.join(self.results_dir, 'logreg_feature_importance.csv'), index=False)

        g = sns.catplot(
            data=feature_importance_df,
            kind='bar',
            x='importance',
            y='feature',
            hue='class',
            col='class',
            col_wrap=3,
            height=4,
            aspect=1.5,
            sharex=False,
            sharey=False
        )
        g.fig.suptitle('Top Features by Class - Logistic Regression', fontsize=16)
        g.fig.subplots_adjust(top=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "figures", "logreg_feature_importance.png"))
        plt.close()

        print("Feature importance saved.")

    def train_bert_model(self, train_encodings, y_train, val_encodings=None, y_val=None,
                         model_name="bert-base-uncased", num_labels=8, epochs=3,
                         batch_size=16, learning_rate=5e-5):
        print(f"\nTraining BERT model: {model_name}...")

        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'].numpy(),
            'attention_mask': train_encodings['attention_mask'].numpy(),
            'labels': y_train
        })

        val_dataset = None
        if val_encodings is not None and y_val is not None:
            val_dataset = Dataset.from_dict({
                'input_ids': val_encodings['input_ids'].numpy(),
                'attention_mask': val_encodings['attention_mask'].numpy(),
                'labels': y_val
            })

        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        training_args = TrainingArguments(
            output_dir=os.path.join(self.models_dir, 'bert_checkpoints'),
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="f1" if val_dataset else None,
            push_to_hub=False,
        )

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            return {
                'accuracy': accuracy_score(labels, preds),
                'f1': f1_score(labels, preds, average='weighted')
            }

        trainer = Trainer(
            model=self.bert_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if val_dataset else None,
            compute_metrics=compute_metrics if val_dataset else None
        )

        trainer.train()
        trainer.save_model(os.path.join(self.models_dir, 'bert_model'))

        print("BERT model saved.")

        if val_dataset:
            eval_results = trainer.evaluate()
            print(f"Validation Results: {eval_results}")

            with open(os.path.join(self.results_dir, 'bert_results.pkl'), 'wb') as f:
                pickle.dump({
                    'model': f'BERT ({model_name})',
                    'accuracy': eval_results['eval_accuracy'],
                    'weighted_f1': eval_results['eval_f1']
                }, f)

            with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
                json.dump({
                    "f1_score": round(eval_results['eval_f1'], 5),
                    "accuracy": round(eval_results['eval_accuracy'], 5)
                }, f)

            val_preds = trainer.predict(val_dataset)
            y_val_pred = np.argmax(val_preds.predictions, axis=1)

            cm = confusion_matrix(y_val, y_val_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - BERT')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(self.results_dir, "figures", "bert_confusion_matrix.png"))
            plt.close()

            print("Confusion matrix and evaluation results saved.")
            return eval_results['eval_accuracy'], eval_results['eval_f1']

        print("Validation data not provided — skipping evaluation.")
        return None, None


if __name__ == "__main__":
    # Placeholder for standalone test
    pass
    


import os
import pickle
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification, Trainer, TrainingArguments
)
from datasets import Dataset

# Force MPS fallback to CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class ModelTrainer:
    def __init__(self, models_dir="models", results_dir="results"):
        self.models_dir = models_dir
        self.results_dir = results_dir

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    def train_logistic_regression(self, X_train, y_train, X_val=None, y_val=None, 
                                  C=1.0, max_iter=1000, class_weight='balanced'):
        print(f"\nTraining Logistic Regression model with C={C}, max_iter={max_iter}...")
        self.log_reg = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42
        )
        self.log_reg.fit(X_train, y_train)

        with open(os.path.join(self.models_dir, 'logistic_model.pkl'), 'wb') as f:
            pickle.dump(self.log_reg, f)

        print("Logistic Regression model saved.")

        if X_val is not None and y_val is not None:
            y_val_pred = self.log_reg.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted')

            print(f"Validation accuracy: {val_accuracy:.4f}")
            print(f"Validation weighted F1 score: {val_f1:.4f}")

            cm = confusion_matrix(y_val, y_val_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - Logistic Regression')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(self.results_dir, "figures", "logreg_confusion_matrix.png"))
            plt.close()

            results = {
                'model': 'Logistic Regression',
                'accuracy': val_accuracy,
                'weighted_f1': val_f1
            }
            with open(os.path.join(self.results_dir, 'logreg_results.pkl'), 'wb') as f:
                pickle.dump(results, f)

            with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
                json.dump({
                    "f1_score": round(val_f1, 5),
                    "accuracy": round(val_accuracy, 5)
                }, f)

            return val_accuracy, val_f1

        print("Validation data not provided — skipping evaluation.")
        return None, None

    def analyze_logreg_features(self, vectorizer, label_encoder, top_n=10):
        print(f"\nAnalyzing top {top_n} features per class...")
        feature_names = vectorizer.get_feature_names_out()
        class_labels = label_encoder.classes_

        feature_importance_df = pd.DataFrame()

        for i, class_label in enumerate(class_labels):
            coef = self.log_reg.coef_[i]
            top_indices = coef.argsort()[-top_n:][::-1]
            top_features = [feature_names[idx] for idx in top_indices]
            top_values = coef[top_indices]
            temp_df = pd.DataFrame({
                'feature': top_features,
                'importance': top_values,
                'class': class_label
            })
            feature_importance_df = pd.concat([feature_importance_df, temp_df])

        feature_importance_df.to_csv(os.path.join(self.results_dir, 'logreg_feature_importance.csv'), index=False)

        g = sns.catplot(
            data=feature_importance_df,
            kind='bar',
            x='importance',
            y='feature',
            hue='class',
            col='class',
            col_wrap=3,
            height=4,
            aspect=1.5,
            sharex=False,
            sharey=False
        )
        g.fig.suptitle('Top Features by Class - Logistic Regression', fontsize=16)
        g.fig.subplots_adjust(top=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "figures", "logreg_feature_importance.png"))
        plt.close()

        print("Feature importance saved.")

    def train_bert_model(self, train_encodings, y_train, val_encodings=None, y_val=None,
                         model_name="bert-base-uncased", num_labels=8, epochs=3,
                         batch_size=16, learning_rate=5e-5):
        print(f"\nTraining BERT model: {model_name}...")

        train_encodings = {k: v.cpu() for k, v in train_encodings.items()}
        if val_encodings:
            val_encodings = {k: v.cpu() for k, v in val_encodings.items()}

        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'].numpy(),
            'attention_mask': train_encodings['attention_mask'].numpy(),
            'labels': y_train
        })

        val_dataset = None
        if val_encodings is not None and y_val is not None:
            val_dataset = Dataset.from_dict({
                'input_ids': val_encodings['input_ids'].numpy(),
                'attention_mask': val_encodings['attention_mask'].numpy(),
                'labels': y_val
            })

        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to("cpu")

        training_args = TrainingArguments(
            output_dir=os.path.join(self.models_dir, 'bert_checkpoints'),
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="f1" if val_dataset else None,
            push_to_hub=False,
            no_cuda=True
        )

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            return {
                'accuracy': accuracy_score(labels, preds),
                'f1': f1_score(labels, preds, average='weighted')
            }

        trainer = Trainer(
            model=self.bert_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if val_dataset else None,
            compute_metrics=compute_metrics if val_dataset else None
        )

        trainer.train()
        trainer.save_model(os.path.join(self.models_dir, 'bert_model'))

        print("BERT model saved.")

        if val_dataset:
            eval_results = trainer.evaluate()
            print(f"Validation Results: {eval_results}")

            with open(os.path.join(self.results_dir, 'bert_results.pkl'), 'wb') as f:
                pickle.dump({
                    'model': f'BERT ({model_name})',
                    'accuracy': eval_results['eval_accuracy'],
                    'weighted_f1': eval_results['eval_f1']
                }, f)

            with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
                json.dump({
                    "f1_score": round(eval_results['eval_f1'], 5),
                    "accuracy": round(eval_results['eval_accuracy'], 5)
                }, f)

            val_preds = trainer.predict(val_dataset)
            y_val_pred = np.argmax(val_preds.predictions, axis=1)

            cm = confusion_matrix(y_val, y_val_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - BERT')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(self.results_dir, "figures", "bert_confusion_matrix.png"))
            plt.close()

            print("Confusion matrix and evaluation results saved.")
            return eval_results['eval_accuracy'], eval_results['eval_f1']

        print("Validation data not provided — skipping evaluation.")
        return None, None


if __name__ == "__main__":
    # Placeholder for standalone test
    pass








    
import os
import pickle
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
)
from datasets import Dataset

# Force MPS fallback to CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class ModelTrainer:
    def __init__(self, models_dir="models", results_dir="results"):
        self.models_dir = models_dir
        self.results_dir = results_dir

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)

    def train_logistic_regression(self, X_train, y_train, X_val=None, y_val=None, 
                                  C=1.0, max_iter=1000, class_weight='balanced'):
        print(f"\nTraining Logistic Regression model with C={C}, max_iter={max_iter}...")
        self.log_reg = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42
        )
        self.log_reg.fit(X_train, y_train)

        with open(os.path.join(self.models_dir, 'logistic_model.pkl'), 'wb') as f:
            pickle.dump(self.log_reg, f)

        print("Logistic Regression model saved.")

        if X_val is not None and y_val is not None:
            y_val_pred = self.log_reg.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted')

            print(f"Validation accuracy: {val_accuracy:.4f}")
            print(f"Validation weighted F1 score: {val_f1:.4f}")

            cm = confusion_matrix(y_val, y_val_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - Logistic Regression')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(self.results_dir, "figures", "logreg_confusion_matrix.png"))
            plt.close()

            results = {
                'model': 'Logistic Regression',
                'accuracy': val_accuracy,
                'weighted_f1': val_f1
            }
            with open(os.path.join(self.results_dir, 'logreg_results.pkl'), 'wb') as f:
                pickle.dump(results, f)

            with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
                json.dump({
                    "f1_score": round(val_f1, 5),
                    "accuracy": round(val_accuracy, 5)
                }, f)

            return val_accuracy, val_f1

        print("Validation data not provided — skipping evaluation.")
        return None, None

    def analyze_logreg_features(self, vectorizer, label_encoder, top_n=10):
        print(f"\nAnalyzing top {top_n} features per class...")
        feature_names = vectorizer.get_feature_names_out()
        class_labels = label_encoder.classes_

        feature_importance_df = pd.DataFrame()

        for i, class_label in enumerate(class_labels):
            coef = self.log_reg.coef_[i]
            top_indices = coef.argsort()[-top_n:][::-1]
            top_features = [feature_names[idx] for idx in top_indices]
            top_values = coef[top_indices]
            temp_df = pd.DataFrame({
                'feature': top_features,
                'importance': top_values,
                'class': class_label
            })
            feature_importance_df = pd.concat([feature_importance_df, temp_df])

        feature_importance_df.to_csv(os.path.join(self.results_dir, 'logreg_feature_importance.csv'), index=False)

        g = sns.catplot(
            data=feature_importance_df,
            kind='bar',
            x='importance',
            y='feature',
            hue='class',
            col='class',
            col_wrap=3,
            height=4,
            aspect=1.5,
            sharex=False,
            sharey=False
        )
        g.fig.suptitle('Top Features by Class - Logistic Regression', fontsize=16)
        g.fig.subplots_adjust(top=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "figures", "logreg_feature_importance.png"))
        plt.close()

        print("Feature importance saved.")

    def train_bert_model(self, train_encodings, y_train, val_encodings=None, y_val=None,
                         model_name="roberta-large", num_labels=8, epochs=10,
                         batch_size=16, learning_rate=5e-5):
        print(f"\nTraining BERT model: {model_name}...")

        train_encodings = {k: v.cpu() for k, v in train_encodings.items()}
        if val_encodings:
            val_encodings = {k: v.cpu() for k, v in val_encodings.items()}

        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'].numpy(),
            'attention_mask': train_encodings['attention_mask'].numpy(),
            'labels': y_train
        })

        val_dataset = None
        if val_encodings is not None and y_val is not None:
            val_dataset = Dataset.from_dict({
                'input_ids': val_encodings['input_ids'].numpy(),
                'attention_mask': val_encodings['attention_mask'].numpy(),
                'labels': y_val
            })

        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to("cpu")

        training_args = TrainingArguments(
            output_dir=os.path.join(self.models_dir, 'bert_checkpoints'),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            no_cuda=True,
            logging_dir=os.path.join(self.results_dir, 'logs'),
            report_to="none"
        )

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            return {
                'accuracy': accuracy_score(labels, preds),
                'f1': f1_score(labels, preds, average='weighted')
            }

        trainer = Trainer(
            model=self.bert_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train()
        trainer.save_model(os.path.join(self.models_dir, 'bert_model'))

        print("BERT model saved.")

        if val_dataset:
            eval_results = trainer.evaluate()
            print(f"Validation Results: {eval_results}")

            with open(os.path.join(self.results_dir, 'bert_results.pkl'), 'wb') as f:
                pickle.dump({
                    'model': f'BERT ({model_name})',
                    'accuracy': eval_results['eval_accuracy'],
                    'weighted_f1': eval_results['eval_f1']
                }, f)

            with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
                json.dump({
                    "f1_score": round(eval_results['eval_f1'], 5),
                    "accuracy": round(eval_results['eval_accuracy'], 5)
                }, f)

            val_preds = trainer.predict(val_dataset)
            y_val_pred = np.argmax(val_preds.predictions, axis=1)

            cm = confusion_matrix(y_val, y_val_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - BERT')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(self.results_dir, "figures", "bert_confusion_matrix.png"))
            plt.close()

            print("Confusion matrix and evaluation results saved.")
            return eval_results['eval_accuracy'], eval_results['eval_f1']

        print("Validation data not provided — skipping evaluation.")
        return None, None


if __name__ == "__main__":
    # Placeholder for standalone test
    pass


'''


#src/model_training.py

import os
import pickle
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
)
from datasets import Dataset

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class ModelTrainer:
    def __init__(self, models_dir="models", results_dir="results"):
        self.models_dir = models_dir
        self.results_dir = results_dir

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)

    def train_logistic_regression(self, X_train, y_train, X_val=None, y_val=None, 
                                  C=1.0, max_iter=1000, class_weight='balanced'):
        print(f"\nTraining Logistic Regression model with C={C}, max_iter={max_iter}...")
        self.log_reg = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42
        )
        self.log_reg.fit(X_train, y_train)

        with open(os.path.join(self.models_dir, 'logistic_model.pkl'), 'wb') as f:
            pickle.dump(self.log_reg, f)

        print("Logistic Regression model saved.")

        if X_val is not None and y_val is not None:
            y_val_pred = self.log_reg.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted')

            print(f"Validation accuracy: {val_accuracy:.4f}")
            print(f"Validation weighted F1 score: {val_f1:.4f}")

            cm = confusion_matrix(y_val, y_val_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - Logistic Regression')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(self.results_dir, "figures", "logreg_confusion_matrix.png"))
            plt.close()

            results = {
                'model': 'Logistic Regression',
                'accuracy': val_accuracy,
                'weighted_f1': val_f1
            }
            with open(os.path.join(self.results_dir, 'logreg_results.pkl'), 'wb') as f:
                pickle.dump(results, f)

            with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
                json.dump({
                    "f1_score": round(val_f1, 5),
                    "accuracy": round(val_accuracy, 5)
                }, f)

            return val_accuracy, val_f1

        print("Validation data not provided — skipping evaluation.")
        return None, None

    def analyze_logreg_features(self, vectorizer, label_encoder, top_n=10):
        print(f"\nAnalyzing top {top_n} features per class...")
        feature_names = vectorizer.get_feature_names_out()
        class_labels = label_encoder.classes_

        feature_importance_df = pd.DataFrame()

        for i, class_label in enumerate(class_labels):
            coef = self.log_reg.coef_[i]
            top_indices = coef.argsort()[-top_n:][::-1]
            top_features = [feature_names[idx] for idx in top_indices]
            top_values = coef[top_indices]
            temp_df = pd.DataFrame({
                'feature': top_features,
                'importance': top_values,
                'class': class_label
            })
            feature_importance_df = pd.concat([feature_importance_df, temp_df])

        feature_importance_df.to_csv(os.path.join(self.results_dir, 'logreg_feature_importance.csv'), index=False)

        g = sns.catplot(
            data=feature_importance_df,
            kind='bar',
            x='importance',
            y='feature',
            hue='class',
            col='class',
            col_wrap=3,
            height=4,
            aspect=1.5,
            sharex=False,
            sharey=False
        )
        g.fig.suptitle('Top Features by Class - Logistic Regression', fontsize=16)
        g.fig.subplots_adjust(top=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "figures", "logreg_feature_importance.png"))
        plt.close()

        print("Feature importance saved.")

    def train_bert_model(self, train_encodings, y_train, val_encodings=None, y_val=None,
                         model_name="roberta-large", num_labels=8, epochs=5,
                         batch_size=16, learning_rate=2e-5):
        print(f"\nTraining BERT model: {model_name} (LR={learning_rate})...")

        train_encodings = {k: v.cpu() for k, v in train_encodings.items()}
        if val_encodings:
            val_encodings = {k: v.cpu() for k, v in val_encodings.items()}

        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'].numpy(),
            'attention_mask': train_encodings['attention_mask'].numpy(),
            'labels': y_train
        })

        val_dataset = None
        if val_encodings is not None and y_val is not None:
            val_dataset = Dataset.from_dict({
                'input_ids': val_encodings['input_ids'].numpy(),
                'attention_mask': val_encodings['attention_mask'].numpy(),
                'labels': y_val
            })

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to("cpu")

        training_args = TrainingArguments(
            output_dir=os.path.join(self.models_dir, 'bert_checkpoints'),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            no_cuda=True,
            logging_dir=os.path.join(self.results_dir, 'logs'),
            report_to="none"
        )

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            return {
                'accuracy': accuracy_score(labels, preds),
                'f1': f1_score(labels, preds, average='weighted')
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train()
        model.save_pretrained(os.path.join(self.models_dir, 'bert_model'))

        if val_dataset:
            eval_results = trainer.evaluate()
            print(f"✅ Final Accuracy: {eval_results['eval_accuracy']:.4f}, F1: {eval_results['eval_f1']:.4f}")

            with open(os.path.join(self.results_dir, 'bert_results.pkl'), 'wb') as f:
                pickle.dump({
                    'model': f'BERT ({model_name})',
                    'accuracy': eval_results['eval_accuracy'],
                    'weighted_f1': eval_results['eval_f1']
                }, f)

            with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
                json.dump({
                    "f1_score": round(eval_results['eval_f1'], 5),
                    "accuracy": round(eval_results['eval_accuracy'], 5)
                }, f)

        return eval_results['eval_accuracy'], eval_results['eval_f1']


if __name__ == "__main__":
    pass