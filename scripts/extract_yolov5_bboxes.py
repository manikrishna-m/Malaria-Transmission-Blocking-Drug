from argparse import ArgumentParser
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from bs4 import BeautifulSoup
from nd2reader import ND2Reader
from skimage import io
from tqdm import tqdm

from tifffile import TiffFileError


def get_image_shape(raw_image_path: Path) -> Tuple[int, int, int]:
    try:
        if raw_image_path.suffix == ".nd2":
            with ND2Reader(raw_image_path) as raw_img:
                return np.array(raw_img).shape
        elif raw_image_path.suffix == ".tif":
            return io.imread(str(raw_image_path)).shape
        else:
            raise ValueError(raw_image_path)
    except TiffFileError as e:
        print(f"Error loading TIFF file: {raw_image_path}. {e}")
        return None


def contains_contours(file: Path) -> Optional[bool]:
    soup = BeautifulSoup(file.read_text(), features="xml")
    classname_tag = soup.find("classname")
    if classname_tag:
        return classname_tag.string == "plugins.kernel.roi.roi2d.ROI2DPolygon"
    else:
        return False


def read_contours(contours_file: Path) -> List[List[Tuple[int, int]]]:
    markup = contours_file.read_text()
    stone_soup = BeautifulSoup(markup, features="xml")
    contour_tags = stone_soup.find_all("points")
    contours = []
    for contour in contour_tags:
        xs = (int(float(x.string)) for x in contour.find_all("pos_x"))
        ys = (int(float(y.string)) for y in contour.find_all("pos_y"))
        shape = list(zip(xs, ys))
        contours.append(shape)
    return contours


def make_bbox(contour: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    max_x = max(x for (x, y) in contour)
    max_y = max(y for (x, y) in contour)
    min_x = min(x for (x, y) in contour)
    min_y = min(y for (x, y) in contour)
    return min_x, min_y, max_x, max_y


def ensure_normalised(x: float) -> float:
    if x <= 0:
        return 0
    elif x > 1:
        return 1
    else:
        return x


def extract_yolov5_bboxes_from_contours(file: Path, image_shape: Tuple[int, int, int]) -> list:
    contours = read_contours(file)
    _, img_height, img_width = image_shape
    result = []
    for contour in contours:
        x0, y0, x1, y1 = make_bbox([(y, x) for (x, y) in contour])
        x_center, y_center = (x0 + x1) / 2, (y0 + y1) / 2
        height = abs(x0 - x1)
        width = abs(y0 - y1)
        x, y, w, h = map(ensure_normalised, (y_center / img_width, x_center / img_height,
                                             width / img_width, height / img_height))
        if w > 0 and h > 0:
            result.append((0, x, y, w, h))
    return result


def extract_yolov5_bboxes(file: Path, image_shape: Tuple[int, int, int]) -> List[Tuple[int, float, float, float, float]]:
    if contains_contours(file):
        return extract_yolov5_bboxes_from_contours(file, image_shape=image_shape)
    else:
        return []


def convert_xml_bboxes(file: Path, input_format: str, output_folder: Path):
    output_folder.mkdir(exist_ok=True, parents=True)
    output_file = output_folder / f"{file.stem}.txt"
    image_file = file.parent / f"{file.stem}.{input_format}"
    try:
        image_shape = get_image_shape(image_file)
        if image_shape is None:
            return
    except FileNotFoundError:
        print(f"Image file not found: {image_file}. Skipping...")
        return
    bboxes = extract_yolov5_bboxes(file, image_shape=image_shape)
    if bboxes:
        labels_file_content = "\n".join(
            " ".join(str(x) for x in bbox)
            for bbox in bboxes
        ) + "\n"
        output_file.write_text(labels_file_content)


def convert_folder(input_folder: Path, input_format: str, output_folder: Path, n_workers: int = 1) -> List[Path]:
    source_files = list(input_folder.glob("**/*.xml"))

    def converter_fn(file):
        return convert_xml_bboxes(file, output_folder=output_folder, input_format=input_format)

    with ThreadPool(processes=n_workers) as pool:
        return list(tqdm(pool.imap(converter_fn, source_files), total=len(source_files)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input-folder", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=cpu_count())
    parser.add_argument("--input-format", type=str, default="tif")

    args = parser.parse_args()
    args.output_folder.mkdir(exist_ok=True, parents=True)
    convert_folder(input_folder=args.input_folder,
                   input_format=args.input_format,
                   output_folder=args.output_folder)
