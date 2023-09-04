import numpy as np
import pickle
import sys

from typing import List, Tuple, Dict
from pathlib import Path
from argparse import ArgumentParser
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from loguru import logger
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import pandas as pd

logger.remove()
logger.add(sys.stdout, level="INFO")

def load_embeddings(file: Path) -> Tuple[List[Path], np.ndarray]:
    embeddings = pickle.loads(file.read_bytes())
    files, vectors = zip(*list(embeddings.items()))
    return files, vectors

def evaluate_clustering(vectors: np.ndarray, clustering) -> Tuple[np.ndarray, float, float, float]:
    clusters = clustering.fit_predict(vectors)
    sil_score = round(silhouette_score(vectors, clusters), 2)
    ch_score = round(calinski_harabasz_score(vectors, clusters), 2)
    db_score = round(davies_bouldin_score(vectors, clusters), 2)
    return clusters, sil_score, ch_score, db_score

def calculate_cluster_intersection(clusters1, clusters2):
    return clusters1 == clusters2

def plot_cluster_sums(cluster_sums, output_dir):
    num_clusters = len(cluster_sums['KMeans'])
    cluster_indices = np.arange(1, num_clusters + 1)
    bar_width = 0.3  # Adjust this value for spacing between bars

    plt.figure(figsize=(14, 6))  # Increase width to accommodate both plots side by side

 # Plot for KMeans
    plt.bar(cluster_indices - bar_width, cluster_sums['KMeans'], label='KMeans', width=bar_width)

    # Plot for Hierarchical
    plt.bar(cluster_indices, cluster_sums['Hierarchical'], label='Hierarchical', width=bar_width)

    # Plot for Common
    plt.bar(cluster_indices + bar_width, cluster_sums['Grouped'], label='Common', width=bar_width)


    plt.xlabel('Cluster')
    plt.ylabel('Sum of Cluster')
    plt.title('Sum of Clusters Comparison')
    plt.xticks(cluster_indices)
    plt.legend()
    plt.tight_layout()

    # Save bar plot as PNG
    plt.savefig(output_dir / 'cluster_sums_comparison.png')
    plt.show()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('embeddings_file', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True, default='data/')
    args = parser.parse_args()

    logger.info("Loading embeddings...")
    files, vectors = load_embeddings(file=args.embeddings_file)

    kmeans_params = {
        'n_clusters': 4,
    }

    agglomerative_params = {
        'n_clusters': 4,
        'linkage': 'ward'
    }

    logger.info(f"Running KMeans Clustering with n_clusters={kmeans_params['n_clusters']}")
    kmeans = KMeans(n_clusters=kmeans_params['n_clusters'], init='k-means++', random_state=0)
    clusters_kmeans, _, _, _ = evaluate_clustering(vectors, kmeans)


    logger.info(f"Running Agglomerative Clustering with n_clusters={agglomerative_params['n_clusters']}, linkage={agglomerative_params['linkage']}")
    agglomerative = AgglomerativeClustering(n_clusters=agglomerative_params['n_clusters'], linkage=agglomerative_params['linkage'])
    clusters_agglomerative, _, _, _ = evaluate_clustering(vectors, agglomerative)

    logger.info("Clustering completed.")

    cluster_mapping1 = dict(zip(files, clusters_kmeans))

    cluster_mapping2 = dict(zip(files, clusters_agglomerative))

    # Convert the dictionary to a pandas DataFrame
    cluster_df = pd.DataFrame({'File': list(cluster_mapping1.keys()),
                               'KMeans Cluster': list(cluster_mapping1.values()),
                               'Agglomerative Cluster': list(cluster_mapping2.values())})

    cluster_sums = {
        'KMeans': [],
        'Hierarchical': [],
        'Grouped': []

    }

    num_clusters = 4

    for cluster in range(0, num_clusters):
        kmeans_sum = np.sum(clusters_kmeans == cluster)
        agglomerative_sum = np.sum(clusters_agglomerative == cluster)
        combined_sum = np.sum((clusters_kmeans == cluster) & (clusters_agglomerative == cluster))
        cluster_sums['KMeans'].append(kmeans_sum)
        cluster_sums['Hierarchical'].append(agglomerative_sum)
        cluster_sums['Grouped'].append(combined_sum)

    print(cluster_sums)

    plot_cluster_sums(cluster_sums, args.output_dir)

