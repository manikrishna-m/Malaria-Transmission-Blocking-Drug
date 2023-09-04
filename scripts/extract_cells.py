from argparse import ArgumentParser
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List, Optional, Tuple, Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm



def extract_bboxes(file: Path) -> Iterable[Tuple[float, float, float, float]]:
    for line in file.read_text().splitlines():
        _, x, y, w, h = (float(i) for i in line.split())
        x0 = x - w / 2
        x1 = x + w / 2
        y0 = y - h / 2
        y1 = y + h / 2
        result = x0, y0, x1, y1
        yield result


def extract_images_from_bboxes(labels_file: Path, images_folder: Path, output_folder: Path, image_extension: str = "jpg"):
    output_folder.mkdir(exist_ok=True, parents=True)
    image: Image.Image = Image.open(images_folder / f"{labels_file.stem}.{image_extension}")
    bboxes = extract_bboxes(labels_file)
    if bboxes:
        for i, bbox in enumerate(bboxes):
            target_folder = output_folder / labels_file.stem
            target_folder.mkdir(exist_ok=True, parents=True)
            x0, y0, x1, y1 = bbox
            width = image.width
            height = image.height
            image.crop((x0 * width, y0 * height, x1 * width, y1 * height)).save(target_folder / f"{i}.{image_extension}")


def extract_cells(labels_folder: Path, images_folder: Path, output_folder: Path, n_workers: int = 1) -> List[Path]:
    labels_files = list(labels_folder.glob("**/*.txt"))

    def extractor_fn(labels_file):
        return extract_images_from_bboxes(labels_file=labels_file, images_folder=images_folder, output_folder=output_folder)

    with ThreadPool(processes=n_workers) as pool:
        return list(tqdm(pool.imap(extractor_fn, labels_files), total=len(labels_files)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--labels-folder", type=Path, required=True)
    parser.add_argument("--images-folder", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=cpu_count())

    args = parser.parse_args()
    args.output_folder.mkdir(exist_ok=True, parents=True)
    extract_cells(
        labels_folder=args.labels_folder,
        images_folder=args.images_folder,
        output_folder=args.output_folder,
        n_workers=args.workers
    )
