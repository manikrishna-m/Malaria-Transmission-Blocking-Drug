import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, List

import torch
import glob
import os
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def get_model() -> torch.nn.Module:
    model = EfficientNet.from_pretrained("efficientnet-b1")
    model.eval()
    return model


def get_transform(width: int, height: int) -> Callable[[Image.Image], torch.Tensor]:
    """Provide a transform that converts image to a square and keeps the cell shape"""
    target_dim = max(width, height)
    w_pad = (target_dim - width) // 2
    h_pad = (target_dim - height) // 2
    transform = transforms.Compose([
        transforms.Pad(padding=(w_pad, h_pad)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform


def get_image_embeddings(image: Path) -> List[float]:
    input_image = Image.open(image).convert("RGB")
    preprocess = get_transform(input_image.width, input_image.height)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    return output[0].cpu().detach().tolist()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--images", type=Path, nargs="+")
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    model = get_model()
    image_paths = []
    for pattern in args.images:
        image_paths.extend(glob.glob(str(pattern)))
    mapping = {str(i): get_image_embeddings(image=i) 
               for i in tqdm(sorted(image_paths))}
    output_file = args.output_file
    output_file.write_bytes(pickle.dumps(mapping))
