import numpy as np
import pickle
import sys

from typing import List, Tuple, Dict
from pathlib import Path
from argparse import ArgumentParser
from tabulate import tabulate
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from kneed import KneeLocator
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
import joblib
import csv

import logging
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

def load_embeddings(file: Path) -> Tuple[List[Path], np.ndarray]:
    embeddings = pickle.loads(file.read_bytes())
    files, vectors = zip(*list(embeddings.items()))
    return files, vectors

def save_cluster_mapping(output_file: Path, cluster_mapping: Dict[Path, int]):
    output_file.write_bytes(pickle.dumps(cluster_mapping))

def run_kmeans_elbow(vectors: np.ndarray, cluster_range: List[int], output_dir: Path):
    logger.info("Starting KMeans Clustering with Elbow Method...")
    
    distortions = []
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(vectors)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 4))
    plt.plot(cluster_range, distortions, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.xticks(cluster_range)
    plt.savefig(output_dir)
    plt.close()

    kl = KneeLocator(cluster_range, distortions, curve='convex', direction='decreasing')
    optimal_clusters = kl.elbow

    logger.info(f"Optimal number of clusters based on Elbow Method: {optimal_clusters}")
    return optimal_clusters

def evaluate_clustering(vectors: np.ndarray, clustering) -> Tuple[np.ndarray, float, float, float]:
    clusters = clustering.fit_predict(vectors)
    sil_score = round(silhouette_score(vectors, clusters), 2)
    ch_score = round(calinski_harabasz_score(vectors, clusters), 2)
    db_score = round(davies_bouldin_score(vectors, clusters), 2)
    return clusters, sil_score, ch_score, db_score

def save_agglomerative_dendrogram(vectors: np.ndarray, linkage_type: str, save_dir: Path):
    linkage_matrix = linkage(vectors, method=linkage_type)
    dendrogram_filename = f"dendrogram_{linkage_type}.png"
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, truncate_mode='level', leaf_font_size=8)
    plt.title(f"Agglomerative Dendrogram ({linkage_type} Linkage)")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(save_dir / dendrogram_filename)
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('embeddings_file', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True, default='data/')
    args = parser.parse_args()

    logger.info("Loading embeddings...")
    files, vectors = load_embeddings(file=args.embeddings_file)

    results = []

    kmeans_params = {
        'n_clusters_case1': list(range(3, 15)),
        'n_clusters_case2': list(range(5, 45, 5)),
        'init': ['k-means++', 'random'],
    }

    agglomerative_params = {
        'n_clusters_complete': list(range(15, 21)),
        'n_clusters_ward': list(range(4, 11)),
        'linkage': ['complete', 'ward']
    }

    dendrogram_params = {
        'linkage': ['complete', 'ward', 'simple', 'average']
    }


    optimal_clusters = run_kmeans_elbow(vectors, kmeans_params['n_clusters_case1'], args.output_dir / "elbow_15.png")
    for init_method in kmeans_params['init']:
        logger.info(f"Running KMeans Clustering with n_clusters={optimal_clusters}, init={init_method}")
        kmeans_case1 = KMeans(n_clusters=optimal_clusters, init=init_method, random_state=0)
        clusters, sil_score, ch_score, db_score = evaluate_clustering(vectors, kmeans_case1)

        result_row = ["KMeans", optimal_clusters, init_method, sil_score, ch_score, db_score]
        results.append(result_row)

        if optimal_clusters == 6:
            clusters_kmeans = clusters 


    optimal_clusters = run_kmeans_elbow(vectors, kmeans_params['n_clusters_case2'], args.output_dir / "elbow_45.png")
    
    for init_method in kmeans_params['init']:
        logger.info(f"Running KMeans Clustering with n_clusters={optimal_clusters}, init={init_method}")
        kmeans_case2 = KMeans(n_clusters=optimal_clusters, init=init_method, random_state=0)
        clusters, sil_score, ch_score, db_score = evaluate_clustering(vectors, kmeans_case2)

        result_row = ["KMeans", optimal_clusters, init_method, sil_score, ch_score, db_score]
        results.append(result_row)

    logger.info("KMeans Clustering completed.")

    # Agglomerative Clustering
    clusters_agglomerative = None


    dendrogram_save_dir = args.output_dir / "dendrograms"
    dendrogram_save_dir.mkdir(exist_ok=True)

    for linkage_type in agglomerative_params['linkage']:
        save_agglomerative_dendrogram(vectors, linkage_type, dendrogram_save_dir)


    for linkage_type in agglomerative_params['linkage']:
        n_clusters_list = agglomerative_params[f'n_clusters_{linkage_type}']

        for n_clusters in n_clusters_list:
            logger.info(f"Running Agglomerative Clustering with n_clusters={n_clusters}, linkage={linkage_type}")

            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
            clusters, sil_score, ch_score, db_score = evaluate_clustering(vectors, clustering)
        
            results.append(["Agglomerative", n_clusters, linkage_type, sil_score, ch_score, db_score])

            if linkage_type == 'ward' and n_clusters == 4:
                clusters_agglomerative = clusters

    logger.info("Agglomerative Clustering completed.")

    cluster_mapping = dict(zip(files, clusters_agglomerative))
    output_file = args.output_dir / "best_agglomerative_model.pkl"
    save_cluster_mapping(output_file, cluster_mapping)

    cluster_mapping = dict(zip(files, clusters_kmeans))
    output_file = args.output_dir / "best_kmeans_model.pkl"
    save_cluster_mapping(output_file, cluster_mapping)

    logger.info("Best models saved.")

    kmeans_cluster_assignments = list(zip(files, clusters_kmeans))
    kmeans_csv_file = args.output_dir / "kmeans_cluster_assignments.csv"
    with open(kmeans_csv_file, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["Filename", "Cluster"])
        csv_writer.writerows(kmeans_cluster_assignments)

    agglomerative_cluster_assignments = list(zip(files, clusters_agglomerative))
    agglomerative_csv_file = args.output_dir / "agglomerative_cluster_assignments.csv"
    with open(agglomerative_csv_file, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["Filename", "Cluster"])
        csv_writer.writerows(agglomerative_cluster_assignments)

    print(tabulate(results, headers=["Model", "Clusters", "Init/Linkage", "Silhouette Score", "Calinski-Harabasz Score", "Davies-Bouldin Score"], tablefmt="grid"))
