from argparse import ArgumentParser
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List, Tuple
from tifffile import TiffFileError

import numpy as np
from skimage import io
from nd2reader import ND2Reader
from PIL import Image
from tqdm import tqdm

def load_raw(raw_image_path: Path) -> np.ndarray:
    try:
        if raw_image_path.suffix == ".nd2":
            with ND2Reader(raw_image_path) as raw_img:
                return np.array(raw_img)
        elif raw_image_path.suffix == ".tif":
            return io.imread(str(raw_image_path))
        else:
            raise ValueError(raw_image_path)
    except TiffFileError as e:
        print(f"Error loading TIFF file: {raw_image_path}. {e}")
        return None
    


def compute_channel_range(channel: np.ndarray, q_low: float, q_high: float) -> Tuple[float, float]:
    low = np.quantile(channel, q=q_low)
    hight = np.quantile(channel, q=q_high)
    return low, hight


def convert_array(img: np.ndarray,
                  ) -> Image.Image:
    # R = 0 for every image
    img_3channel = np.stack((np.zeros_like(img[0]), *img, ))

    # Reorder channels from RBG to RGB
    img_3channel[1:3] = img_3channel[2:0:-1]

    # Clip G
    green_min, green_max = compute_channel_range(
        img_3channel[1], q_low=0.90, q_high=0.9995)
    img_3channel[1] = np.clip(img_3channel[1], green_min, green_max)

    # Clip B
    blue_min, blue_max = compute_channel_range(
        img_3channel[2], q_low=0.95, q_high=0.9995)
    img_3channel[2] = np.clip(img_3channel[2], blue_min, blue_max)

    # Convert to [3 * w * h] for PIL
    result = np.rollaxis(img_3channel, 0, 3).astype(float)

    # Scale to [0,1]
    result[:, :, 1] = (result[:, :, 1] - result[:, :, 1].min()) / \
        (result[:, :, 1].max() - result[:, :, 1].min())
    result[:, :, 2] = (result[:, :, 2] - result[:, :, 2].min()) / \
        (result[:, :, 2].max() - result[:, :, 2].min())

    # Scale back to [0,255] and create PIL image
    converted = Image.fromarray(np.uint8(result * 255), mode="RGB")
    return converted


def convert_image(raw_image_path: Path, output_folder: Path, extension: str) -> Path:
    source_image = load_raw(raw_image_path)
    if source_image is None:
        print(f"Skipping conversion for file: {raw_image_path}")
        return None
    
    converted_image = convert_array(source_image)
    output_folder.mkdir(exist_ok=True, parents=True)
    target_path = output_folder / f"{raw_image_path.stem}.{extension}"
    converted_image.save(target_path)
    return target_path


def convert_folder(input_folder: Path, input_format: str, output_folder: Path, extension: str, n_workers: int = 1) -> List[Path]:
    source_files = list(input_folder.glob(f"**/*.{input_format}"))

    def converter_fn(file):
        return convert_image(file, output_folder=output_folder, extension=extension)

    with ThreadPool(processes=n_workers) as pool:
        return list(tqdm(pool.imap(converter_fn, source_files), total=len(source_files)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input-folder", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--output-format", type=str, default="jpg")
    parser.add_argument("--input-format", type=str, default="tif")
    parser.add_argument("--workers", type=int, default=cpu_count())
    args = parser.parse_args()
    args.output_folder.mkdir(exist_ok=True, parents=True)
    convert_folder(input_folder=args.input_folder,
                   input_format=args.input_format,
                   output_folder=args.output_folder,
                   extension=args.output_format,
                   n_workers=args.workers)
