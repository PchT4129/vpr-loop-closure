import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ImageFolderDataset, get_default_transform
from src.models import ResNet18FeatureExtractor


def extract_features(image_dir: str | Path, batch_size: int = 16):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ImageFolderDataset(
        root_dir=image_dir,
        transform=get_default_transform(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    model = ResNet18FeatureExtractor(pretrained=True).to(device)
    model.eval()

    all_features = []
    all_paths = []

    with torch.no_grad():
        for images, paths in tqdm(loader, desc=f"Extracting {image_dir}"):
            images = images.to(device)

            features = model(images)

            all_features.append(features.cpu())
            all_paths.extend(paths)

    all_features = torch.cat(all_features, dim=0)

    return all_features, all_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    features, paths = extract_features(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "features": features,
            "paths": paths,
        },
        output_path,
    )

    print(f"Saved {len(paths)} features to {output_path}")
    print(f"Feature shape: {features.shape}")


if __name__ == "__main__":
    main()