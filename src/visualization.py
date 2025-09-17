#src/visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import umap
import hdbscan
import os
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.colors as mcolors
from collections import Counter


class TopicVisualizer:
    def __init__(self, data_dir="data", models_dir="models", results_dir="results"):
        """Initialize the TopicVisualizer class."""
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # Create directories if they don't exist
        os.makedirs(os.path.join(self.results_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "embeddings"), exist_ok=True)
    
    def load_data(self):
        """Load processed data."""
        print("Loading processed data...")
        
        self.train_df = pd.read_csv(os.path.join(self.data_dir, "processed", "train_processed.csv"))
        
        print(f"Loaded {len(self.train_df)} training examples.")
        
        return self.train_df
    
    def extract_bert_embeddings(self, model_name="bert-base-uncased", max_length=128):
        """Extract embeddings using BERT."""
        print(f"\nExtracting BERT embeddings using {model_name}...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Set model to evaluation mode
        model.eval()
        
        # Create embeddings in batches to avoid memory issues
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(self.train_df), batch_size):
            print(f"Processing batch {i // batch_size + 1}/{(len(self.train_df) - 1) // batch_size + 1}")
            
            batch_texts = self.train_df['Interview Text'].iloc[i:i+batch_size].tolist()
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                
                # Use [CLS] token embeddings
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(batch_embeddings)
        
        # Combine batches
        self.embeddings = np.vstack(embeddings)
        
        # Normalize embeddings
        self.embeddings = normalize(self.embeddings)
        
        # Save embeddings
        np.save(os.path.join(self.results_dir, "embeddings", "bert_embeddings.npy"), self.embeddings)
        
        print(f"Extracted and saved {self.embeddings.shape} BERT embeddings.")
        
        return self.embeddings
    
    def reduce_dimensions(self, method="umap", n_components=2):
        """Reduce dimensionality of embeddings for visualization."""
        print(f"\nReducing dimensions using {method}...")
        
        if method == "umap":
            # Use UMAP for dimension reduction
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            self.reduced_embeddings = reducer.fit_transform(self.embeddings)
        elif method == "tsne":
            # Use t-SNE for dimension reduction
            reducer = TSNE(n_components=n_components, random_state=42)
            self.reduced_embeddings = reducer.fit_transform(self.embeddings)
        elif method == "svd":
            # Use TruncatedSVD for dimension reduction
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
            self.reduced_embeddings = reducer.fit_transform(self.embeddings)
        
        # Save reduced embeddings
        np.save(os.path.join(self.results_dir, "embeddings", f"{method}_embeddings.npy"), self.reduced_embeddings)
        
        print(f"Reduced dimensions to {self.reduced_embeddings.shape}.")
        
        return self.reduced_embeddings
    
    def find_optimal_clusters(self, max_clusters=20):
        """Find optimal number of clusters using silhouette score."""
        print("\nFinding optimal number of clusters...")
        
        silhouette_scores = []
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.embeddings)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(self.embeddings, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            
            print(f"Clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.4f}")
        
        # Find optimal number of clusters
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        plt.axvline(x=optimal_clusters, color='r', linestyle='--')
        plt.title('Silhouette Score Method For Optimal k')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "figures", "optimal_clusters.png"))
        plt.close()
        
        print(f"Optimal number of clusters: {optimal_clusters}")
        
        return optimal_clusters
    
    def cluster_topics(self, n_clusters=None):
        """Cluster data into topics."""
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters()
        
        print(f"\nClustering into {n_clusters} topics...")
        
        # Use KMeans for clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        
        # Add cluster labels to DataFrame
        self.train_df['cluster'] = self.cluster_labels
        
        # Save clustered data
        self.train_df.to_csv(os.path.join(self.results_dir, "embeddings", "clustered_data.csv"), index=False)
        
        print(f"Data clustered into {n_clusters} topics and saved.")
        
        return self.cluster_labels
    
    def visualize_topics(self, interactive=True):
        """Visualize topics using dimensionality reduction."""
        print("\nVisualizing topics...")
        
        # Create DataFrame for visualization
        viz_df = pd.DataFrame({
            'x': self.reduced_embeddings[:, 0],
            'y': self.reduced_embeddings[:, 1],
            'cluster': self.cluster_labels,
            'label': self.train_df['Labels'],
            'text': self.train_df['Interview Text'].apply(lambda x: x[:100] + '...' if len(x) > 100 else x)
        })
        
        if interactive:
            # Create interactive plot with Plotly
            fig = px.scatter(
                viz_df,
                x='x',
                y='y',
                color='cluster',
                hover_data=['label', 'text'],
                labels={'cluster': 'Topic Cluster'},
                title='Topic Clusters Visualization'
            )
            
            # Save as HTML
            fig.write_html(os.path.join(self.results_dir, "figures", "topic_clusters_interactive.html"))
            print(f"Interactive topic visualization saved to {os.path.join(self.results_dir, 'figures', 'topic_clusters_interactive.html')}")
        
        # Create static plot with Matplotlib
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='x', y='y', hue='cluster', data=viz_df, palette='viridis')
        plt.title('Topic Clusters Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend(title='Topic Cluster')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "figures", "topic_clusters.png"))
        plt.close()
        
        print(f"Static topic visualization saved to {os.path.join(self.results_dir, 'figures', 'topic_clusters.png')}")
    
    def analyze_clusters(self):
        """Analyze the content of each cluster."""
        print("\nAnalyzing clusters...")
        
        # Group by cluster
        cluster_summaries = []
        
        for cluster_id in range(len(set(self.cluster_labels))):
            # Get texts in cluster
            cluster_texts = self.train_df[self.train_df['cluster'] == cluster_id]['cleaned_text']
            
            # Combine texts
            combined_text = ' '.join(cluster_texts)
            
            # Get most common words
            vectorizer = TfidfVectorizer(max_features=10)
            tfidf_matrix = vectorizer.fit_transform([combined_text])
            
            # Get top terms
            feature_names = vectorizer.get_feature_names_out()
            top_indices = tfidf_matrix.toarray()[0].argsort()[-10:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            
            # Get label distribution
            label_dist = self.train_df[self.train_df['cluster'] == cluster_id]['Labels'].value_counts().to_dict()
            
            # Add to summaries
            cluster_summaries.append({
                'cluster_id': cluster_id,
                'size': len(cluster_texts),
                'top_terms': top_terms,
                'label_distribution': label_dist
            })
        
        # Convert to DataFrame
        self.cluster_analysis = pd.DataFrame(cluster_summaries)
        
        # Save analysis
        self.cluster_analysis.to_csv(os.path.join(self.results_dir, "embeddings", "cluster_analysis.csv"), index=False)
        
        # Generate cluster summary document
        with open(os.path.join(self.results_dir, "embeddings", "cluster_summary.md"), 'w') as f:
            f.write("# Topic Cluster Analysis\n\n")
            f.write(f"Total number of clusters: {len(set(self.cluster_labels))}\n\n")
            
            for _, row in self.cluster_analysis.iterrows():
                f.write(f"## Cluster {row['cluster_id']}\n\n")
                f.write(f"**Size:** {row['size']} interviews\n\n")
                f.write("**Top Terms:**\n")
                for term in row['top_terms']:
                    f.write(f"- {term}\n")
                f.write("\n**Label Distribution:**\n")
                for label, count in row['label_distribution'].items():
                    f.write(f"- {label}: {count} interviews\n")
                f.write("\n---\n")
    
    def create_wordclouds(self):
        """Create wordclouds for each cluster."""
        print("\nCreating wordclouds for each cluster...")
        
        # Colors for wordclouds
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        # Create a directory for wordclouds
        os.makedirs(os.path.join(self.results_dir, "figures", "wordclouds"), exist_ok=True)
        
        # Create wordcloud for each cluster
        for cluster_id in range(len(set(self.cluster_labels))):
            # Get texts in cluster
            cluster_texts = self.train_df[self.train_df['cluster'] == cluster_id]['cleaned_text']
            
            # Combine texts
            combined_text = ' '.join(cluster_texts)
            
            # Create wordcloud
            wordcloud = WordCloud(
                background_color='white',
                max_words=100,
                width=800,
                height=400,
                contour_width=3,
                contour_color=colors[cluster_id % len(colors)],
                colormap='viridis'
            ).generate(combined_text)
            
            # Plot wordcloud
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Top Words in Cluster {cluster_id}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "figures", "wordclouds", f"cluster_{cluster_id}_wordcloud.png"))
            plt.close()
        
        print(f"Wordclouds saved to {os.path.join(self.results_dir, 'figures', 'wordclouds')}")
    
    def plot_label_distribution(self):
        """Plot distribution of labels within clusters."""
        print("\nPlotting label distribution within clusters...")
        
        # Create a pivot table of cluster vs label
        pivot_df = pd.crosstab(self.train_df['cluster'], self.train_df['Labels'])
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt="d")
        plt.title('Distribution of Labels within Clusters')
        plt.xlabel('Label')
        plt.ylabel('Cluster')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "figures", "cluster_label_distribution.png"))
        plt.close()
        
        print(f"Label distribution plot saved to {os.path.join(self.results_dir, 'figures', 'cluster_label_distribution.png')}")
    
    def generate_3d_visualization(self):
        """Generate a 3D visualization of clusters."""
        print("\nGenerating 3D visualization...")
        
        # Reduce to 3 dimensions
        reducer = umap.UMAP(n_components=3, random_state=42)
        embeddings_3d = reducer.fit_transform(self.embeddings)
        
        # Create DataFrame
        viz_3d_df = pd.DataFrame({
            'x': embeddings_3d[:, 0],
            'y': embeddings_3d[:, 1],
            'z': embeddings_3d[:, 2],
            'cluster': self.cluster_labels,
            'label': self.train_df['Labels'],
            'text': self.train_df['Interview Text'].apply(lambda x: x[:100] + '...' if len(x) > 100 else x)
        })
        
        # Create 3D plot
        fig = px.scatter_3d(
            viz_3d_df,
            x='x',
            y='y',
            z='z',
            color='cluster',
            hover_data=['label', 'text'],
            labels={'cluster': 'Topic Cluster'},
            title='3D Topic Clusters Visualization'
        )
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3'
            ),
            width=900,
            height=800
        )
        
        # Save as HTML
        fig.write_html(os.path.join(self.results_dir, "figures", "3d_topic_clusters.html"))
        
        print(f"3D visualization saved to {os.path.join(self.results_dir, 'figures', '3d_topic_clusters.html')}")
    
    def visualize_embeddings_for_streamlit(self, embeddings=None, labels=None, texts=None):
        """Create visualization data for Streamlit."""
        if embeddings is None:
            embeddings = self.reduced_embeddings
        
        if labels is None:
            labels = self.train_df['Labels']
        
        if texts is None:
            texts = self.train_df['Interview Text']
        
        # Create DataFrame for visualization
        viz_df = pd.DataFrame({
            'x': embeddings[:, 0],
            'y': embeddings[:, 1],
            'label': labels,
            'text': texts.apply(lambda x: x[:100] + '...' if len(x) > 100 else x)
        })
        
        return viz_df
    
    def run_full_pipeline(self, model_name="bert-base-uncased", n_clusters=None):
        """Run the full topic visualization pipeline."""
        self.load_data()
        self.extract_bert_embeddings(model_name=model_name)
        self.reduce_dimensions(method="umap")
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters()
        self.cluster_topics(n_clusters=n_clusters)
        self.visualize_topics(interactive=True)
        self.analyze_clusters()
        self.create_wordclouds()
        self.plot_label_distribution()
        self.generate_3d_visualization()
        
        return {
            'num_clusters': len(set(self.cluster_labels)),
            'cluster_analysis': self.cluster_analysis,
            'reduced_embeddings': self.reduced_embeddings,
            'cluster_labels': self.cluster_labels
        }


class DataVisualizer:
    def __init__(self, data_dir="data", results_dir="results"):
        """Initialize the DataVisualizer class."""
        self.data_dir = data_dir
        self.results_dir = results_dir
        
        # Create directories if they don't exist
        os.makedirs(os.path.join(self.results_dir, "figures"), exist_ok=True)
    
    def load_data(self):
        """Load data."""
        print("Loading data...")
        
        self.train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        
        print(f"Loaded {len(self.train_df)} training examples.")
        
        return self.train_df
    
    def visualize_class_distribution(self):
        """Visualize class distribution."""
        print("\nVisualizing class distribution...")
        
        # Count labels
        label_counts = self.train_df['Labels'].value_counts().sort_index()
        
        # Plot bar chart
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=label_counts.index, y=label_counts.values)
        plt.title('Distribution of Interview Categories')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Add count labels on top of bars
        for i, count in enumerate(label_counts.values):
            ax.text(i, count + 5, str(count), ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "figures", "class_distribution.png"))
        plt.close()
        
        print(f"Class distribution visualization saved to {os.path.join(self.results_dir, 'figures', 'class_distribution.png')}")
        
        return label_counts
    
    def visualize_sports_distribution(self):
        """Visualize distribution of sports."""
        print("\nVisualizing sports distribution...")
        
        # Count sports
        sports_counts = self.train_df['Sports'].value_counts()
        
        # Plot pie chart
        plt.figure(figsize=(10, 10))
        plt.pie(sports_counts.values, labels=sports_counts.index, autopct='%1.1f%%', startangle=90, shadow=True)
        plt.title('Distribution of Sports')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "figures", "sports_distribution.png"))
        plt.close()
        
        print(f"Sports distribution visualization saved to {os.path.join(self.results_dir, 'figures', 'sports_distribution.png')}")
        
        return sports_counts
    
    def visualize_interview_types(self):
        """Visualize distribution of interview types."""
        print("\nVisualizing interview types distribution...")
        
        # Count interview types
        type_counts = self.train_df['Type'].value_counts()
        
        # Plot horizontal bar chart
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(y=type_counts.index, x=type_counts.values, orient='h')
        plt.title('Distribution of Interview Types')
        plt.xlabel('Count')
        plt.ylabel('Interview Type')
        
        # Add count labels
        for i, count in enumerate(type_counts.values):
            ax.text(count + 5, i, str(count), va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "figures", "interview_types.png"))
        plt.close()
        
        print(f"Interview types visualization saved to {os.path.join(self.results_dir, 'figures', 'interview_types.png')}")
        
        return type_counts
    
    def visualize_text_length_distribution(self):
        """Visualize distribution of text lengths."""
        print("\nVisualizing text length distribution...")
        
        # Calculate text lengths
        self.train_df['text_length'] = self.train_df['Interview Text'].apply(len)
        
        # Plot histogram
        plt.figure(figsize=(12, 6))
        sns.histplot(self.train_df['text_length'], bins=50, kde=True)
        plt.title('Distribution of Interview Text Lengths')
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Count')
        plt.axvline(self.train_df['text_length'].mean(), color='r', linestyle='--', label=f'Mean: {self.train_df["text_length"].mean():.0f}')
        plt.axvline(self.train_df['text_length'].median(), color='g', linestyle='--', label=f'Median: {self.train_df["text_length"].median():.0f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "figures", "text_length_distribution.png"))
        plt.close()
        
        print(f"Text length distribution visualization saved to {os.path.join(self.results_dir, 'figures', 'text_length_distribution.png')}")
        
        # Text length statistics
        length_stats = {
            'mean': self.train_df['text_length'].mean(),
            'median': self.train_df['text_length'].median(),
            'min': self.train_df['text_length'].min(),
            'max': self.train_df['text_length'].max()
        }
        
        return length_stats
    
    def visualize_label_by_sports(self):
        """Visualize relationship between labels and sports."""
        print("\nVisualizing labels by sports...")
        
        # Create cross-tabulation
        label_sport_df = pd.crosstab(self.train_df['Labels'], self.train_df['Sports'])
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(label_sport_df, annot=True, cmap="YlGnBu", fmt="d")
        plt.title('Distribution of Labels by Sports')
        plt.xlabel('Sports')
        plt.ylabel('Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "figures", "label_by_sports.png"))
        plt.close()
        
        print(f"Labels by sports visualization saved to {os.path.join(self.results_dir, 'figures', 'label_by_sports.png')}")
        
        return label_sport_df
    
    def visualize_label_by_type(self):
        """Visualize relationship between labels and interview types."""
        print("\nVisualizing labels by interview types...")
        
        # Create cross-tabulation
        label_type_df = pd.crosstab(self.train_df['Labels'], self.train_df['Type'])
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(label_type_df, annot=True, cmap="YlGnBu", fmt="d")
        plt.title('Distribution of Labels by Interview Types')
        plt.xlabel('Interview Type')
        plt.ylabel('Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "figures", "label_by_type.png"))
        plt.close()
        
        print(f"Labels by interview types visualization saved to {os.path.join(self.results_dir, 'figures', 'label_by_type.png')}")
        
        return label_type_df
    
    def run_full_pipeline(self):
        """Run the full data visualization pipeline."""
        self.load_data()
        self.visualize_class_distribution()
        self.visualize_sports_distribution()
        self.visualize_interview_types()
        self.visualize_text_length_distribution()
        self.visualize_label_by_sports()
        self.visualize_label_by_type()
        
        return {
            'class_distribution': self.train_df['Labels'].value_counts().to_dict(),
            'sports_distribution': self.train_df['Sports'].value_counts().to_dict(),
            'type_distribution': self.train_df['Type'].value_counts().to_dict()
        }