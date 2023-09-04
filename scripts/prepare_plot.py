import pickle
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Lock, cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List, Tuple
import time

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

lock = Lock()

def load_file_contents(file: Path) -> Tuple[List[Path], np.ndarray]:
    data = pickle.loads(file.read_bytes())
    files, points = zip(*list(data.items()))
    return files, np.array(points)

def create_figure_2d(points: np.ndarray, clusters: np.ndarray, images: List[Path], limit_images: int = 500) -> go.FigureWidget:
    x = points[:, 0]
    y = points[:, 1]
    fig = px.scatter(x=x, y=y, color=clusters, opacity=0.75, size_max=5)
    np.random.seed(42)
    logger.info("Adding images to layout")
    
    num_workers = min(cpu_count() * 2, limit_images)  
    
    with ThreadPool(num_workers) as pool:
        shuffled_data = np.random.permutation(list(zip(images, x, y)))[:limit_images]
        batch_size = 250 
        for batch_start in range(0, len(shuffled_data), batch_size):
            batch = shuffled_data[batch_start:batch_start + batch_size]
            _ = list(tqdm(
                pool.imap(
                    partial(add_image_to_layout, lock=lock, fig=fig),
                    batch
                ),
                total=len(batch))
            )
    return fig

def add_image_to_layout(image_x_y: Tuple[Path, float, float], fig: go.FigureWidget, lock: Lock):
    image, x, y = image_x_y
    pil_image = Image.open(image)
    lock.acquire()
    fig.add_layout_image(dict(
        source=pil_image,
        x=x,
        y=y,
        xref="x",
        yref="y",
        sizex=2,
        sizey=2,
        opacity=1,
        xanchor="center", yanchor="middle",
        layer="below",
    ))
    lock.release()

def create_figure(points: np.ndarray, clusters: np.ndarray, images: List[Path], limit_images: int) -> go.FigureWidget:
    fig = create_figure_2d(points=points, clusters=clusters, images=images, limit_images=limit_images)
    return fig

def prepare_plot(points_file: Path, clusters_file: Path, output_file: Path, limit_images: int):
    points_files, points = load_file_contents(file=points_file)
    _, clusters = load_file_contents(file=clusters_file)
    start_time = time.time()  # Record start time
    fig: go.FigureWidget = create_figure(points=points, clusters=clusters, images=points_files, limit_images=limit_images)
    output_file.write_text(fig.to_json())
    
    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time
    logger.info(f"Time taken to prepare and generate the plot: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--points', type=Path, required=True)
    parser.add_argument('--clusters', type=Path, required=True)
    parser.add_argument('--output-file', type=Path, required=True)
    parser.add_argument('--limit-images', type=int, required=False, default=500)
    args = parser.parse_args()
    prepare_plot(
        output_file=args.output_file,
        points_file=args.points,
        clusters_file=args.clusters,
        limit_images=args.limit_images)
