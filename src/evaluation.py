'''import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class ModelEvaluator:
    def __init__(self, models_dir="models", results_dir="results", data_dir="data"):
        """Initialize the ModelEvaluator class."""
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.data_dir = data_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
    
    def load_models(self):
        """Load trained models and required components."""
        print("Loading models and components...")
        
        # Load logistic regression model
        with open(os.path.join(self.models_dir, 'logistic_model.pkl'), 'rb') as f:
            self.log_reg = pickle.load(f)
        
        # Load TF-IDF vectorizer
        with open(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        # Load label encoder
        with open(os.path.join(self.models_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load BERT model and tokenizer
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(self.models_dir, 'bert_model')
        )
        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.models_dir, 'bert_tokenizer')
        )
        
        print("Models and components loaded successfully.")
    
    def compare_models(self):
        """Compare the performance of trained models."""
        print("\nComparing model performance...")
        
        # Load evaluation results
        with open(os.path.join(self.results_dir, 'logreg_results.pkl'), 'rb') as f:
            logreg_results = pickle.load(f)
        
        with open(os.path.join(self.results_dir, 'bert_results.pkl'), 'rb') as f:
            bert_results = pickle.load(f)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame([
            {
                'Model': 'Logistic Regression',
                'Accuracy': logreg_results['accuracy'],
                'Weighted F1': logreg_results['weighted_f1']
            },
            {
                'Model': 'BERT',
                'Accuracy': bert_results['accuracy'],
                'Weighted F1': bert_results['weighted_f1']
            }
        ])
        
        # Save comparison
        comparison_df.to_csv(os.path.join(self.results_dir, 'model_comparison.csv'), index=False)
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        sns.barplot(x='Model', y='Accuracy', data=comparison_df)
        plt.title('Accuracy Comparison')
        plt.ylim(0, 1)
        
        # Plot F1 score
        plt.subplot(1, 2, 2)
        sns.barplot(x='Model', y='Weighted F1', data=comparison_df)
        plt.title('Weighted F1 Score Comparison')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "figures", "model_comparison.png"))
        plt.close()
        
        print(f"Model comparison saved to {os.path.join(self.results_dir, 'model_comparison.csv')}")
        print(f"Comparison plot saved to {os.path.join(self.results_dir, 'figures', 'model_comparison.png')}")
        
        # Determine best model
        if bert_results['weighted_f1'] > logreg_results['weighted_f1']:
            print("BERT model performs better based on weighted F1 score.")
            return "bert"
        else:
            print("Logistic Regression model performs better based on weighted F1 score.")
            return "logreg"
    
    def predict_test_data(self, model_choice="bert"):
        """Generate predictions on test data using the selected model."""
        print(f"\nGenerating predictions using {model_choice} model...")
        
        # Load test data
        test_df = pd.read_csv(os.path.join(self.data_dir, "processed", "test_processed.csv"))
        
        if model_choice == "logreg":
            # Transform test data using TF-IDF
            X_test_tfidf = self.tfidf_vectorizer.transform(test_df['cleaned_text'])
            
            # Predict
            y_pred = self.log_reg.predict(X_test_tfidf)
            
        elif model_choice == "bert":
            # Tokenize test data
            test_encodings = self.bert_tokenizer(
                test_df['Interview Text'].tolist(),
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            # Create Dataset
            from datasets import Dataset
            test_dataset = Dataset.from_dict({
                'input_ids': test_encodings['input_ids'].numpy(),
                'attention_mask': test_encodings['attention_mask'].numpy()
            })
            
            # Set up inference
            self.bert_model.eval()
            
            # Predict
            from transformers import Trainer
            trainer = Trainer(model=self.bert_model)
            predictions = trainer.predict(test_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)
        
        # Convert numerical predictions back to labels
        predicted_labels = self.label_encoder.inverse_transform(y_pred)
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'ID': test_df['ID'],
            'Labels': predicted_labels
        })
        
        # Save submission
        submission_df.to_csv(os.path.join(self.results_dir, 'submission.csv'), index=False)
        
        print(f"Predictions saved to {os.path.join(self.results_dir, 'submission.csv')}")
        
        return submission_df
    
    def evaluate_with_ground_truth(self, ground_truth_path):
        """Evaluate predictions against ground truth."""
        print("\nEvaluating predictions against ground truth...")
        
        # Load ground truth
        ground_truth_df = pd.read_csv(ground_truth_path)
        
        # Load predictions
        predictions_df = pd.read_csv(os.path.join(self.results_dir, 'submission.csv'))
        
        # Merge DataFrames
        merged_df = pd.merge(ground_truth_df, predictions_df, on='ID', suffixes=('_true', '_pred'))
        
        # Encode labels
        y_true = self.label_encoder.transform(merged_df['Labels_true'])
        y_pred = self.label_encoder.transform(merged_df['Labels_pred'])
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Create results dictionary
        results = {
            'accuracy': float(accuracy),
            'f1_score': float(weighted_f1)
        }
        
        # Save results
        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=self.label_encoder.classes_, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(self.results_dir, 'classification_report.csv'))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - Final Predictions vs Ground Truth')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "figures", "final_confusion_matrix.png"))
        plt.close()
        
        print(f"Evaluation results saved to {os.path.join(self.results_dir, 'results.json')}")
        print(f"Classification report saved to {os.path.join(self.results_dir, 'classification_report.csv')}")
        print(f"Confusion matrix saved to {os.path.join(self.results_dir, 'figures', 'final_confusion_matrix.png')}")
        
        print(f"Final accuracy: {accuracy:.4f}")
        print(f"Final weighted F1 score: {weighted_f1:.4f}")
        
        return results






if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate models and generate predictions.")
    parser.add_argument('--mode', choices=['dev', 'submission'], default='dev',
                        help="Run mode: 'dev' evaluates using val.csv, 'submission' only outputs submission.csv.")
    args = parser.parse_args()

    evaluator = ModelEvaluator()
    evaluator.load_models()

    best_model = evaluator.compare_models(val_data_path="data/val.csv")
    evaluator.predict_test_data(model_choice=best_model)

    if args.mode == 'dev':
        evaluator.evaluate_with_ground_truth("data/val.csv")
    else:
        print("Submission mode: evaluation skipped. Submit 'results/submission.csv' to leaderboard.")







if __name__ == "__main__":
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load models
    evaluator.load_models()
    
    # Compare models and get best model
    best_model = evaluator.compare_models()
    
    # Generate predictions with best model
    evaluator.predict_test_data(model_choice=best_model)
    
    # Evaluate against ground truth
    evaluator.evaluate_with_ground_truth("data/ground_truth.csv")
    
    
    
    
import pandas as pd
import numpy as np
import pickle
import os
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from datasets import Dataset

class ModelEvaluator:
    def __init__(self, models_dir="models", results_dir="results", data_dir="data"):
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.data_dir = data_dir

        os.makedirs(results_dir, exist_ok=True)

    def load_models(self):
        print("Loading models and components...")
        with open(os.path.join(self.models_dir, 'logistic_model.pkl'), 'rb') as f:
            self.log_reg = pickle.load(f)
        with open(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        with open(os.path.join(self.models_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)

        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(self.models_dir, 'bert_model')
        )
        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.models_dir, 'bert_tokenizer')
        )
        print("Models loaded successfully.")

    def compare_models(self):
        print("\nComparing model performance...")
        with open(os.path.join(self.results_dir, 'logreg_results.pkl'), 'rb') as f:
            logreg_results = pickle.load(f)
        with open(os.path.join(self.results_dir, 'bert_results.pkl'), 'rb') as f:
            bert_results = pickle.load(f)

        comparison_df = pd.DataFrame([
            {
                'Model': 'Logistic Regression',
                'Accuracy': logreg_results['accuracy'],
                'Weighted F1': logreg_results['weighted_f1']
            },
            {
                'Model': 'BERT',
                'Accuracy': bert_results['accuracy'],
                'Weighted F1': bert_results['weighted_f1']
            }
        ])
        comparison_df.to_csv(os.path.join(self.results_dir, 'model_comparison.csv'), index=False)

        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        sns.barplot(x='Model', y='Accuracy', data=comparison_df)
        plt.title('Accuracy Comparison')
        plt.ylim(0, 1)
        plt.subplot(1, 2, 2)
        sns.barplot(x='Model', y='Weighted F1', data=comparison_df)
        plt.title('Weighted F1 Score Comparison')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "figures", "model_comparison.png"))
        plt.close()

        print("Model comparison saved.")
        return "bert" if bert_results['weighted_f1'] > logreg_results['weighted_f1'] else "logreg"

    def predict_test_data(self, model_choice="bert"):
        print(f"\nGenerating predictions using {model_choice} model...")
        test_df = pd.read_csv(os.path.join(self.data_dir, "processed", "test_processed.csv"))

        if model_choice == "logreg":
            X_test = self.tfidf_vectorizer.transform(test_df['cleaned_text'])
            y_pred = self.log_reg.predict(X_test)
        else:
            encodings = self.bert_tokenizer(
                test_df['Interview Text'].tolist(),
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            test_dataset = Dataset.from_dict({
                'input_ids': encodings['input_ids'].cpu().numpy(),
                'attention_mask': encodings['attention_mask'].cpu().numpy()
            })
            self.bert_model.eval()
            trainer = Trainer(model=self.bert_model)
            predictions = trainer.predict(test_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)

        predicted_labels = self.label_encoder.inverse_transform(y_pred)
        submission_df = pd.DataFrame({
            'ID': test_df['ID'],
            'Labels': predicted_labels
        })
        submission_df.to_csv(os.path.join(self.results_dir, 'submission.csv'), index=False)
        print("Submission saved.")
        return submission_df

    def evaluate_with_ground_truth(self, ground_truth_path):
        print("\nEvaluating against ground truth...")
        ground_truth_df = pd.read_csv(ground_truth_path)
        predictions_df = pd.read_csv(os.path.join(self.results_dir, 'submission.csv'))

        merged_df = pd.merge(ground_truth_df, predictions_df, on='ID', suffixes=('_true', '_pred'))

        y_true = self.label_encoder.transform(merged_df['Labels_true'])
        y_pred = self.label_encoder.transform(merged_df['Labels_pred'])

        accuracy = accuracy_score(y_true, y_pred)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')

        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump({
                "f1_score": round(weighted_f1, 5),
                "accuracy": round(accuracy, 5)
            }, f, indent=2)

        report = classification_report(y_true, y_pred, target_names=self.label_encoder.classes_, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(os.path.join(self.results_dir, 'classification_report.csv'))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - Final Predictions')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(self.results_dir, "figures", "final_confusion_matrix.png"))
        plt.close()

        print(f"Evaluation complete. Accuracy: {accuracy:.4f}, F1: {weighted_f1:.4f}")
        return {"accuracy": accuracy, "f1_score": weighted_f1}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['dev', 'submission'], default='dev',
                        help="Run mode: 'dev' (with ground truth) or 'submission' (no evaluation)")
    args = parser.parse_args()

    evaluator = ModelEvaluator()
    evaluator.load_models()
    best_model = evaluator.compare_models()
    evaluator.predict_test_data(model_choice=best_model)

    if args.mode == 'dev':
        evaluator.evaluate_with_ground_truth("data/ground_truth.csv")
    else:
        print("Submission mode: evaluation skipped. Submit 'results/submission.csv'.")
'''


#src/evaluation.py
import pandas as pd
import numpy as np
import pickle
import os
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

class ModelEvaluator:
    def __init__(self, models_dir="models", results_dir="results", data_dir="data"):
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.data_dir = data_dir
        os.makedirs(results_dir, exist_ok=True)

    def load_models(self):
        print("Loading models and components...")
        with open(os.path.join(self.models_dir, 'logistic_model.pkl'), 'rb') as f:
            self.log_reg = pickle.load(f)
        with open(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        with open(os.path.join(self.models_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)

        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(self.models_dir, 'bert_model')
        ).to("cpu")  # ✅ Force CPU
        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.models_dir, 'bert_tokenizer')
        )
        print("Models loaded successfully.")

    def compare_models(self):
        print("\nComparing model performance...")
        with open(os.path.join(self.results_dir, 'logreg_results.pkl'), 'rb') as f:
            logreg_results = pickle.load(f)
        with open(os.path.join(self.results_dir, 'bert_results.pkl'), 'rb') as f:
            bert_results = pickle.load(f)

        comparison_df = pd.DataFrame([
            {'Model': 'Logistic Regression', 'Accuracy': logreg_results['accuracy'], 'Weighted F1': logreg_results['weighted_f1']},
            {'Model': 'BERT', 'Accuracy': bert_results['accuracy'], 'Weighted F1': bert_results['weighted_f1']}
        ])
        comparison_df.to_csv(os.path.join(self.results_dir, 'model_comparison.csv'), index=False)

        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        sns.barplot(x='Model', y='Accuracy', data=comparison_df)
        plt.title('Accuracy Comparison')
        plt.ylim(0, 1)
        plt.subplot(1, 2, 2)
        sns.barplot(x='Model', y='Weighted F1', data=comparison_df)
        plt.title('Weighted F1 Score Comparison')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "figures", "model_comparison.png"))
        plt.close()

        print("Model comparison saved.")
        return "bert" if bert_results['weighted_f1'] > logreg_results['weighted_f1'] else "logreg"

    def predict_test_data(self, model_choice="bert"):
        print(f"\nGenerating predictions using {model_choice} model...")
        test_df = pd.read_csv(os.path.join(self.data_dir, "processed", "test_processed.csv"))

        if model_choice == "logreg":
            X_test = self.tfidf_vectorizer.transform(test_df['cleaned_text'])
            y_pred = self.log_reg.predict(X_test)
        else:
            encodings = self.bert_tokenizer(
                test_df['Interview Text'].tolist(),
                truncation=True,
                padding='max_length',
                max_length=128
            )

            test_dataset = Dataset.from_dict({
                'input_ids': np.array(encodings['input_ids']),
                'attention_mask': np.array(encodings['attention_mask'])
            })

            training_args = TrainingArguments(
                output_dir="./tmp",
                per_device_eval_batch_size=16,
                no_cuda=True  # ✅ Ensure CPU usage
            )

            trainer = Trainer(model=self.bert_model, args=training_args)
            predictions = trainer.predict(test_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)

        predicted_labels = self.label_encoder.inverse_transform(y_pred)
        submission_df = pd.DataFrame({
            'ID': test_df['ID'],
            'Labels': predicted_labels
        })
        submission_df.to_csv(os.path.join(self.results_dir, 'submission.csv'), index=False)
        print("Submission saved.")
        return submission_df

    def evaluate_on_validation_set(self):
        print("\nEvaluating on validation set...")

        val_df = pd.read_csv(os.path.join(self.data_dir, "processed", "val_processed.csv"))
        best_model = self.compare_models()

        if best_model == "logreg":
            X_val = self.tfidf_vectorizer.transform(val_df['cleaned_text'])
            y_pred = self.log_reg.predict(X_val)
        else:
            encodings = self.bert_tokenizer(
                val_df['Interview Text'].tolist(),
                truncation=True,
                padding='max_length',
                max_length=128
            )

            val_dataset = Dataset.from_dict({
                'input_ids': np.array(encodings['input_ids']),
                'attention_mask': np.array(encodings['attention_mask'])
            })

            training_args = TrainingArguments(
                output_dir="./tmp_eval",
                per_device_eval_batch_size=16,
                no_cuda=True
            )

            trainer = Trainer(model=self.bert_model, args=training_args)
            predictions = trainer.predict(val_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)

        y_true = self.label_encoder.transform(val_df['Labels'])
        y_pred = self.label_encoder.transform(self.label_encoder.inverse_transform(y_pred))

        accuracy = accuracy_score(y_true, y_pred)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')

        print(f"✅ Validation Accuracy: {accuracy:.4f}, F1 Score: {weighted_f1:.4f}")

        report = classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        pd.DataFrame(report).transpose().to_csv(
            os.path.join(self.results_dir, 'val_classification_report.csv')
        )

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - Validation Set')
        plt.savefig(os.path.join(self.results_dir, "figures", "val_confusion_matrix.png"))
        plt.close()

        return {"accuracy": accuracy, "f1_score": weighted_f1}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['dev', 'submission'], default='dev',
                        help="Run mode: 'dev' (evaluate on val) or 'submission' (no evaluation)")
    args = parser.parse_args()

    evaluator = ModelEvaluator()
    evaluator.load_models()
    best_model = evaluator.compare_models()
    evaluator.predict_test_data(model_choice=best_model)

    if args.mode == 'dev':
        evaluator.evaluate_on_validation_set()
    else:
        print("Submission mode: evaluation skipped. Submit 'results/submission.csv'.")