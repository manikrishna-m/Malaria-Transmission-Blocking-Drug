# Data Processing and Visualization Pipeline

This repository contains a data processing and visualization pipeline for [BioImages](https://www.ebi.ac.uk/biostudies/bioimages) dataset.

## Prerequisites

- Python 3.6
- Required Python packages (install using `pip install requirements.txt`):
- Download the dataset from [BioImages](https://www.ebi.ac.uk/biostudies/bioimages) and organize it in the following folder structure:
  - `data/src/` (raw data) (Please place the downloaded data in this directory)

## Scripts

1. Convert raw images to JPG format:
   python scripts/convert_raw.py --input-folder data/src/ --output-folder data/images/ --output-format jpg

2. Extract YOLOv5 bounding boxes:
   python scripts/extract_yolov5_bboxes.py --input-folder data/src/ --output-folder data/labels/

3. Extract cells from images:
   python scripts/extract_cells.py --labels-folder data/labels/ --images-folder data/images/ --output-folder data/extracted/

4. Extract embeddings from the extracted cell images:
   python scripts/extract_embeddings.py --images data/extracted/*/*.jpg --output-file data/embeddings.pkl

5. Reduce dimensionality of embeddings:
   python scripts/reduce_dimensionality.py data/embeddings.pkl --output-dir data/

6. Compute clusters from reduced-dimensional embeddings:
   python scripts/compute_clusters.py data/embeddings.pkl --output-dir data/

7. Find common cells in clusters:
   python scripts/commom_cells_in_clusters.py data/embeddings.pkl --output-dir data/

8. Export data to CSV:
   python scripts/export_data_csv.py --clusters data/best_agglomerative_model.pkl --points data/pca.pkl data/kpca.pkl data/tsne_2d.pkl data/tsne_3d.pkl --output-dir data/

9. Prepare data for visualization:
   python scripts/prepare_plot.py --points data/tsne_2d.pkl --clusters data/best_agglomerative_model.pkl --output-file data/plot.json --limit-images 10000

10. Visualize embeddings:
   python scripts/visualize_embeddings.py data/plot.json
