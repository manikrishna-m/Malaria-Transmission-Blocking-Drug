import numpy as np
import pickle
import sys

from typing import List, Tuple
from pathlib import Path
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram

import logging
from loguru import logger

logger.remove()  # Remove the default logger
logger.add(sys.stdout, level="INFO")  # Add a console logger

def load_embeddings(file: Path) -> Tuple[List[Path], np.ndarray]:
    embeddings = pickle.loads(file.read_bytes())
    files, vectors = zip(*list(embeddings.items()))
    return files, vectors

def save_dendrogram(data: np.ndarray, dendrogram_file: Path, linkage_type: str):
    linkage_matrix = linkage(data, method=linkage_type)
    plt.figure(figsize=(12, 6))
    plt.title(f"Dendrogram ({linkage_type} linkage)")
    dendrogram(linkage_matrix, orientation='top')
    plt.savefig(dendrogram_file)
    plt.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('embeddings_file', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True, default='data/dendrograms/')
    args = parser.parse_args()

    logger.info("Loading embeddings...")
    _, vectors = load_embeddings(file=args.embeddings_file)

    dendrogram_linkages = ['ward', 'complete', 'centroid', 'median']

    logger.info("Saving Dendrograms for Different Linkage Methods...")
    
    for linkage_type in dendrogram_linkages:
        logger.info(f"Saving dendrogram for linkage: {linkage_type}")
        
        dendrogram_file = args.output_dir / f"dendrogram_{linkage_type}.png"
        save_dendrogram(vectors, dendrogram_file, linkage_type)

    logger.info("Dendrograms saved for different linkage methods.")
