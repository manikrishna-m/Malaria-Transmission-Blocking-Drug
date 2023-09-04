import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple
import warnings

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA

def load_embeddings(file: Path) -> Tuple[List[Path], np.ndarray]:
    embeddings = pickle.loads(file.read_bytes())
    files, vectors = zip(*list(embeddings.items()))
    return files, vectors

def apply_tSNE_2D(vectors: np.ndarray) -> np.ndarray:
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1, init='pca', learning_rate='auto')
    transformed_data = tsne.fit_transform(vectors)
    return transformed_data

def apply_tSNE_3D(vectors: np.ndarray) -> np.ndarray:
    tsne = TSNE(n_components=3, random_state=42, n_jobs=-1, init='pca', learning_rate='auto')
    transformed_data = tsne.fit_transform(vectors)
    return transformed_data

def apply_kernel_pca(vectors: np.ndarray) -> np.ndarray:
    kpca = KernelPCA(n_components=2, kernel='rbf', random_state=42)
    transformed_data = kpca.fit_transform(vectors)
    return transformed_data

def apply_pca(vectors: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2, random_state=42)
    transformed_data = pca.fit_transform(vectors)
    return transformed_data

def reduce_dimensionality(embeddings_file: Path, output_dir: Path):
    files, vectors = load_embeddings(file=embeddings_file)

    methods = {
        'tsne_2d': apply_tSNE_2D,
        'tsne_3d': apply_tSNE_3D,
        'pca': apply_pca,
        'kpca': apply_kernel_pca
    }

    for method, reduction_function in methods.items():
        transformed_data = reduction_function(np.array(vectors))  # Convert vectors to NumPy array
        points_mapping = dict(zip(files, transformed_data))
        output_file = output_dir / f"{method}.pkl"
        output_file.write_bytes(pickle.dumps(points_mapping))
        print(f"{method.upper()} - Dimensionality reduction completed and saved to {output_file}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('embeddings_file', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    # Ignore warnings
    warnings.filterwarnings("ignore")

    reduce_dimensionality(
        embeddings_file=args.embeddings_file,
        output_dir=args.output_dir)
