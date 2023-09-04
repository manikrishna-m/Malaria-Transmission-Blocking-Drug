from argparse import ArgumentParser
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import albumentations as A
from albumentations.augmentations import bbox_utils

_old_check = bbox_utils.check_bbox

def self_healing_bbox_check(bbox):
    """Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums
    
    Automatically changes bboxes to (0, 1] range
    """
    bbox=list(bbox)
    for i in range(4):
      if (bbox[i]<0) :
        bbox[i]=1e-5
      elif (bbox[i]>1) :
        bbox[i]=1
    bbox=tuple(bbox)
    _old_check(bbox)

bbox_utils.check_bbox = self_healing_bbox_check

def unstar(func):
    """Returns a function that expects args to be passed as tuple"""
    def wrapped(args: Tuple, **kwargs):
        return func(*args, **kwargs)
    return wrapped


def read_image(file: Path) -> np.ndarray:
    image = cv2.imread(str(file))
    return image


def read_bboxes(file: Path) -> List[List[float]]:
    lines = file.read_text().splitlines()
    return [
        [float(i) for i in line.split()][1:] + ['0']
        for line in lines
    ]


def save_augmentation(transformed: dict, image_file: Path, labels_file: Path) -> Tuple[Path, Path]:
    cv2.imwrite(str(image_file), transformed["image"])
    lines = [f"{bbox[-1]} {' '.join(map(str, bbox[:-1]))}" for bbox in transformed["bboxes"]]
    labels_file.write_text("\n".join(lines) + "\n")

def get_transforms() -> Dict[str, A.Compose]:
    result = {}
    for rotation in [0, 90, 180, 270]:
        for do_flip in [0, 1]:
            key = f"{rotation}.{do_flip}"
            result[key] = A.Compose([
                A.HorizontalFlip(p=do_flip),
                A.Rotate(limit=(rotation, rotation), p=1)
            ], bbox_params=A.BboxParams(format="yolo"))
    return result

def augment_image(
        image_file: Path,
        labels_file: Path,
        image_output_folder: Path,
        labels_output_folder: Path
):
    transforms = get_transforms()
    image = read_image(image_file)
    bboxes = read_bboxes(labels_file)
    for transform_key, transform in transforms.items():
        transformed = transform(image=image, bboxes=bboxes)
        image_output_file = image_output_folder/f"{image_file.stem}.{transform_key}{image_file.suffix}"
        labels_output_file = labels_output_folder/f"{labels_file.stem}.{transform_key}{labels_file.suffix}"
        save_augmentation(
            transformed=transformed,
            image_file=image_output_file,
            labels_file=labels_output_file
        )


def augment_images(
        input_folder: Path,
        labels_folder: Path,
        image_output_folder: Path,
        labels_output_folder: Path,
        image_extension: str = "png"):
    images = list(input_folder.glob(f"**/*.{image_extension}"))
    labels = [labels_folder/f"{image.stem}.txt" for image in images]
    image_output_folder.mkdir(parents=True, exist_ok=True)
    labels_output_folder.mkdir(parents=True, exist_ok=True)

    args = [(img_file, labels_file) for img_file, labels_file in zip(images, labels) if labels_file.exists()]

    with ThreadPool(processes=cpu_count()) as pool:
        _ = list(
            tqdm(
                pool.imap(
                    unstar(
                        partial(
                            augment_image,
                            image_output_folder=image_output_folder,
                            labels_output_folder=labels_output_folder
                        ),
                    ),
                    args
                ),
                total=len(args)
            )
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image-input-folder", required=True, type=Path)
    parser.add_argument("--image-output-folder", required=True, type=Path)
    parser.add_argument("--labels-input-folder", required=True, type=Path)
    parser.add_argument("--labels-output-folder", required=True, type=Path)
    parser.add_argument("--image-extension", required=False,
                        default="png", type=str)
    args = parser.parse_args()
    augment_images(
        input_folder=args.image_input_folder,
        labels_folder=args.labels_input_folder,
        image_output_folder=args.image_output_folder,
        labels_output_folder=args.labels_output_folder,
        image_extension=args.image_extension)
