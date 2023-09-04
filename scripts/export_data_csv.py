import pickle
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

def load_file_contents(file: Path) -> dict:
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_clusters(clusters_file: Path, points_files: List[Path], output_dir: Path):
    clusters_data = load_file_contents(clusters_file)

    for index, points_file in enumerate(points_files):
        points_data = load_file_contents(points_file)

        data = defaultdict(dict)
        for key, cluster in clusters_data.items():
            if key in points_data:
                data[key]["cluster"] = cluster
                point = points_data[key]
                if len(point) == 3:  # Three-dimensional data
                    data[key]["point_x"], data[key]["point_y"], data[key]["point_z"] = point
                elif len(point) == 2:  # Two-dimensional data
                    data[key]["point_x"], data[key]["point_y"] = point

        df: pd.DataFrame = pd.DataFrame.from_dict(data, orient="index") \
            .rename_axis('cell_location') \
            .reset_index() \
            .assign(cell_location=lambda df: df.cell_location.apply(Path)) \
            .assign(parent_image=lambda df: df.cell_location.apply(lambda path: path.parent.name))

        if "point_z" in df.columns:
            # Create a 3D scatter plot
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            scatter = None
            for cluster_id, cluster_data in df.groupby("cluster"):
                scatter = ax.scatter(cluster_data["point_x"], cluster_data["point_y"], cluster_data["point_z"], label=f"Cluster {cluster_id}")
            ax.set_xlabel("Point X")
            ax.set_ylabel("Point Y")
            ax.set_zlabel("Point Z")
            plt.title(f"Clusters for {points_file.stem}")
            plt.legend()

            angles = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]

            # Save images from different angles
            for angle in angles:
                ax.view_init(elev=30, azim=angle)
                output_file = output_dir / f"{points_file.stem}_cluster_plot_{angle}deg.png"
                plt.savefig(output_file)
                print(f"Image saved: {output_file}")

            plt.close()
        else:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            scatter = None
            for cluster_id, cluster_data in df.groupby("cluster"):
                scatter = ax.scatter(cluster_data["point_x"], cluster_data["point_y"], label=f"Cluster {cluster_id}")
            ax.set_xlabel("Point X")
            ax.set_ylabel("Point Y")
            plt.title(f"Clusters for {points_file.stem}")
            plt.legend()
            plt.savefig(output_dir / f"{points_file.stem}_cluster_plot.png")
            plt.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--clusters', type=Path, required=True)
    parser.add_argument('--points', type=Path, nargs=4, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    plot_clusters(
        clusters_file=args.clusters,
        points_files=args.points,
        output_dir=args.output_dir)
