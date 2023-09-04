import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple


def get_labelled_images(
        images_folder: Path,
        labels_folder: Path,
        images_format: str) -> List[Path]:
    stem_mapping = defaultdict(dict)
    images = images_folder.glob(f"*.{images_format}")
    labels = labels_folder.glob("*.txt")

    for image in images:
        stem_mapping[image.stem]['image'] = image

    for label in labels:
        stem_mapping[label.stem]['label'] = label

    return [stem["image"] for stem in stem_mapping.values() if len(stem) == 2]


def split_images(images: List[Path], train_size: float, random_seed: int = 42) -> Tuple[List[Path], List[Path]]:
    # need to account for augmentations - to prevent data leak
    image_groups = defaultdict(set)
    for image in images:
        image_key = image.name.split(".")[0]
        image_groups[image_key].add(image)

    image_keys = list(image_groups.keys())

    random.seed(random_seed)
    random.shuffle(image_keys)
    split_point = int(len(image_keys) * train_size)
    train_keys = image_keys[:split_point]
    test_keys = image_keys[split_point:]
    train_images = [image for key in train_keys for image in image_groups[key]]
    test_images = [image for key in test_keys for image in image_groups[key]]
    return train_images, test_images

def create_train_test_files(
        images_folder: Path,
        labels_folder: Path,
        output_folder: Path,
        images_format: str,
        train_size: float) -> Tuple[Path, Path]:
    labelled_images = get_labelled_images(
        images_folder=images_folder,
        images_format=images_format,
        labels_folder=labels_folder)
    train, test = split_images(labelled_images, train_size=train_size)

    train_contents = "\n".join(str(p) for p in train)
    test_contents = "\n".join(str(p) for p in test)

    train_file = (output_folder/"train.txt")
    test_file = (output_folder/"test.txt")

    train_file.write_text(train_contents)
    test_file.write_text(test_contents)

    return train_file, test_file


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--images-folder", type=Path, required=True)
    parser.add_argument("--labels-folder", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--images-format", type=str, required=True)
    parser.add_argument("--train-size", type=float, required=True)

    args = parser.parse_args()
    create_train_test_files(
        images_folder=args.images_folder,
        labels_folder=args.labels_folder,
        output_folder=args.output_folder,
        images_format=args.images_format,
        train_size=args.train_size)
