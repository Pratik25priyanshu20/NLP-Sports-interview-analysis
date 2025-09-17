'''import os
import argparse
import json
import pandas as pd
from datetime import datetime
import time

# Import project modules
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineering
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator
from src.visualization import DataVisualizer, TopicVisualizer
from src.text_generation import TextGenerator

def setup_directories():
    """Set up the project directories structure."""
    directories = [
        "data/processed",
        "models",
        "results/figures",
        "results/embeddings",
        "streamlit/components"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Directory structure set up successfully.")

def run_data_preprocessing(args):
    """Run the data preprocessing pipeline."""
    print("\n" + "="*50)
    print("Starting Data Preprocessing")
    print("="*50)
    
    preprocessor = DataPreprocessor(
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )
    
    if args.load_raw:
        preprocessor.load_raw_data()
    else:
        preprocessor.load_data()
    
    preprocessor.clean_text()
    preprocessor.remove_stopwords()
    preprocessor.tokenize()
    preprocessor.save_processed_data()
    
    print("Data preprocessing completed successfully.")
    return preprocessor

def run_feature_engineering(args, preprocessor=None):
    """Run the feature engineering pipeline."""
    print("\n" + "="*50)
    print("Starting Feature Engineering")
    print("="*50)
    
    engineer = FeatureEngineering(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )
    
    if preprocessor is None:
        engineer.load_processed_data()
    else:
        engineer.train_df = preprocessor.train_df
        engineer.val_df = preprocessor.val_df
        engineer.test_df = preprocessor.test_df
    
    engineer.create_tfidf_features()
    engineer.create_bert_embeddings()
    engineer.save_features()
    
    print("Feature engineering completed successfully.")
    return engineer

def run_model_training(args, engineer=None):
    """Run the model training pipeline."""
    print("\n" + "="*50)
    print("Starting Model Training")
    print("="*50)
    
    trainer = ModelTrainer(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )
    
    if engineer is None:
        trainer.load_features()
    else:
        trainer.X_train_tfidf = engineer.X_train_tfidf
        trainer.X_val_tfidf = engineer.X_val_tfidf
        trainer.y_train = engineer.train_df['Labels']
        trainer.y_val = engineer.val_df['Labels']
    
    trainer.train_logistic_regression()
    trainer.train_random_forest()
    trainer.train_bert_classifier()
    trainer.save_models()
    
    print("Model training completed successfully.")
    return trainer

def run_model_evaluation(args, trainer=None):
    """Run the model evaluation pipeline."""
    print("\n" + "="*50)
    print("Starting Model Evaluation")
    print("="*50)
    
    evaluator = ModelEvaluator(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )
    
    evaluator.load_models()
    evaluator.load_test_data()
    evaluator.evaluate_all_models()
    evaluator.generate_classification_report()
    evaluator.plot_confusion_matrix()
    evaluator.save_predictions()
    evaluator.create_submission_file()
    evaluator.save_results_json()
    
    print("Model evaluation completed successfully.")
    return evaluator

def run_visualization(args):
    """Run the data visualization pipeline."""
    print("\n" + "="*50)
    print("Starting Data Visualization")
    print("="*50)
    
    # General data visualization
    data_viz = DataVisualizer(
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )
    data_viz.run_full_pipeline()
    
    # Topic visualization
    topic_viz = TopicVisualizer(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )
    
    # Only run full topic visualization if specified
    if args.full_topic_viz:
        topic_viz.run_full_pipeline(n_clusters=args.n_clusters)
    
    print("Data visualization completed successfully.")
    return data_viz, topic_viz

def run_text_generation(args):
    """Run the text generation pipeline."""
    print("\n" + "="*50)
    print("Starting Text Generation")
    print("="*50)
    
    generator = TextGenerator(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )
    
    generator.load_data()
    generator.create_example_responses()
    generator.save_examples()
    
    if args.interactive_generation:
        print("\nEnter a category and question to generate a response (or 'quit' to exit):")
        
        while True:
            try:
                category = input("\nEnter category (Game Strategy, Player Performance, etc.): ").strip()
                if category.lower() == 'quit':
                    break
                
                question = input("Enter question: ").strip()
                if question.lower() == 'quit':
                    break
                
                response = generator.generate_response(category, question)
                print("\nGenerated response:")
                print("-" * 50)
                print(response)
                print("-" * 50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    print("Text generation completed successfully.")
    return generator

def main():
    """Main function to run the entire pipeline."""
    parser = argparse.ArgumentParser(description="Chasuite Competition Pipeline")
    
    # Directory arguments
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing data files")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save/load models")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")
    
    # Pipeline control arguments
    parser.add_argument("--load_raw", action="store_true", help="Load raw data instead of processed data")
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--skip_feature_engineering", action="store_true", help="Skip feature engineering step")
    parser.add_argument("--skip_training", action="store_true", help="Skip model training step")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip model evaluation step")
    parser.add_argument("--skip_visualization", action="store_true", help="Skip visualization step")
    parser.add_argument("--skip_text_generation", action="store_true", help="Skip text generation step")
    
    # Model and visualization parameters
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", help="BERT model to use")
    parser.add_argument("--full_topic_viz", action="store_true", help="Run full topic visualization")
    parser.add_argument("--n_clusters", type=int, default=None, help="Number of clusters for topic modeling")
    parser.add_argument("--interactive_generation", action="store_true", help="Enable interactive text generation")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Track start time
    start_time = time.time()
    
    # Run the pipeline
    preprocessor = None
    engineer = None
    trainer = None
    evaluator = None
    
    # Step 1: Data Preprocessing
    if not args.skip_preprocessing:
        preprocessor = run_data_preprocessing(args)
    
    # Step 2: Feature Engineering
    if not args.skip_feature_engineering:
        engineer = run_feature_engineering(args, preprocessor)
    
    # Step 3: Model Training
    if not args.skip_training:
        trainer = run_model_training(args, engineer)
    
    # Step 4: Model Evaluation
    if not args.skip_evaluation:
        evaluator = run_model_evaluation(args, trainer)
    
    # Step 5: Visualization
    if not args.skip_visualization:
        data_viz, topic_viz = run_visualization(args)
    
    # Step 6: Text Generation
    if not args.skip_text_generation:
        generator = run_text_generation(args)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*50)
    print(f"Pipeline completed in {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    print("="*50)
    
    # Summary
    print("\nResults Summary:")
    
    if not args.skip_evaluation and evaluator is not None:
        with open(os.path.join(args.results_dir, "results.json"), 'r') as f:
            results = json.load(f)
        
        print(f"Accuracy: {results['accuracy']:.5f}")
        print(f"F1 Score: {results['f1_score']:.5f}")
    
    if not args.skip_visualization and 'topic_viz' in locals() and args.full_topic_viz:
        if hasattr(topic_viz, 'cluster_labels'):
            print(f"Number of topic clusters found: {len(set(topic_viz.cluster_labels))}")
    
    print("\nFiles saved:")
    print(f"- Submission file: {os.path.join(args.results_dir, 'submission.csv')}")
    print(f"- Results JSON: {os.path.join(args.results_dir, 'results.json')}")
    print(f"- Visualizations: {os.path.join(args.results_dir, 'figures')}")

if __name__ == "__main__":
    main()
    '''
    
    
import os
import argparse
import json
import pandas as pd
import time


from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineering
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator
from src.visualization import DataVisualizer, TopicVisualizer
from src.text_generation import TextGenerator

def setup_directories():
    directories = [
        "data/processed",
        "models",
        "results/figures",
        "results/embeddings",
        "streamlit/components"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Directory structure set up successfully.")

def run_data_preprocessing(args):
    print("\n" + "="*50)
    print("Starting Data Preprocessing")
    print("="*50)

    preprocessor = DataPreprocessor(
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )

    preprocessor.load_data()
    preprocessor.explore_data(preprocessor.train_df)
    preprocessor.preprocess_data()
    preprocessor.visualize_class_distribution()
    preprocessor.create_word_clouds()
    preprocessor.analyze_token_frequency()

    print("Data preprocessing completed successfully.")
    return preprocessor

def run_feature_engineering(args, preprocessor=None):
    print("\n" + "="*50)
    print("Starting Feature Engineering")
    print("="*50)

    engineer = FeatureEngineering(
        data_dir=args.data_dir,
        models_dir=args.models_dir
    )

    if preprocessor:
        engineer.train_df = preprocessor.train_df
        engineer.val_df = preprocessor.val_df
        engineer.test_df = preprocessor.test_df
    else:
        engineer.load_processed_data()

    engineer.create_tfidf_features()
    engineer.encode_labels()
    engineer.prepare_bert_inputs(model_name=args.bert_model)  # ✅ Apply model name here

    print("Feature engineering completed successfully.")
    return engineer

def run_model_training(args, engineer=None):
    print("\n" + "="*50)
    print("Starting Model Training")
    print("="*50)

    trainer = ModelTrainer(
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )

    from joblib import load
    import pickle

    # Load feature/label encodings
    with open(os.path.join(args.models_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)

    trainer.train_logistic_regression(
        engineer.X_train_tfidf, engineer.y_train,
        engineer.X_val_tfidf, engineer.y_val
    )

    trainer.analyze_logreg_features(
        engineer.tfidf_vectorizer, label_encoder
    )

    trainer.train_bert_model(
        train_encodings=engineer.train_encodings,
        y_train=engineer.y_train,
        val_encodings=engineer.val_encodings,
        y_val=engineer.y_val,
        model_name=args.bert_model  # ✅ Pass custom model name to training too
    )

    print("Model training completed successfully.")
    return trainer

def run_model_evaluation(args):
    print("\n" + "="*50)
    print("Starting Model Evaluation")
    print("="*50)

    evaluator = ModelEvaluator(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )

    evaluator.load_models()
    best_model = evaluator.compare_models()
    evaluator.predict_test_data(model_choice=best_model)

    evaluator.evaluate_on_validation_set()

    print("Model evaluation completed successfully.")
    return evaluator

def run_visualization(args):
    print("\n" + "="*50)
    print("Starting Data Visualization")
    print("="*50)

    data_viz = DataVisualizer(
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )
    data_viz.run_full_pipeline()

    topic_viz = TopicVisualizer(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )

    if args.full_topic_viz:
        topic_viz.run_full_pipeline(n_clusters=args.n_clusters)

    print("Data visualization completed successfully.")
    return data_viz, topic_viz

def run_text_generation(args):
    print("\n" + "="*50)
    print("Starting Text Generation")
    print("="*50)

    generator = TextGenerator(
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )

    generator.load_category_examples(os.path.join(args.data_dir, "processed", "train_processed.csv"))
    generator.generate_example_responses()
    generator.write_ethical_reflection()

    print("Text generation completed successfully.")
    return generator

def main():
    parser = argparse.ArgumentParser(description="Chasuite Competition Pipeline")

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--results_dir", type=str, default="results")

    parser.add_argument("--skip_preprocessing", action="store_true")
    parser.add_argument("--skip_feature_engineering", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--skip_visualization", action="store_true")
    parser.add_argument("--skip_text_generation", action="store_true")

    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")  # ✅ Make it configurable
    parser.add_argument("--full_topic_viz", action="store_true")
    parser.add_argument("--n_clusters", type=int, default=None)

    args = parser.parse_args()

    setup_directories()
    start_time = time.time()

    preprocessor = None
    engineer = None
    trainer = None
    evaluator = None

    if not args.skip_preprocessing:
        preprocessor = run_data_preprocessing(args)

    if not args.skip_feature_engineering:
        engineer = run_feature_engineering(args, preprocessor)

    if not args.skip_training:
        trainer = run_model_training(args, engineer)

    if not args.skip_evaluation:
        evaluator = run_model_evaluation(args)

    if not args.skip_visualization:
        run_visualization(args)

    if not args.skip_text_generation:
        run_text_generation(args)

    elapsed_time = time.time() - start_time
    h, r = divmod(elapsed_time, 3600)
    m, s = divmod(r, 60)

    print("\n" + "="*50)
    print(f"Pipeline completed in {int(h):02}:{int(m):02}:{int(s):02}")
    print("="*50)

    if evaluator:
        with open(os.path.join(args.results_dir, "results.json"), "r") as f:
            results = json.load(f)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")

    print("\nFiles saved:")
    print(f"- Submission: {os.path.join(args.results_dir, 'submission.csv')}")
    print(f"- Results:    {os.path.join(args.results_dir, 'results.json')}")
    print(f"- Visuals:    {os.path.join(args.results_dir, 'figures')}")

if __name__ == "__main__":
    main()