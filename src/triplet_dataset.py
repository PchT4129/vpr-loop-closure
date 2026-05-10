import random
import re
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset

from src.dataset import IMAGE_EXTENSIONS, get_default_transform


def image_index_from_path(path: Path) -> int:
    match = re.search(r"Image(\d+)\.jpg$", path.name)
    if match is None:
        raise ValueError(f"Could not parse image index from path: {path}")

    return int(match.group(1))


def find_images(root_dir: str | Path) -> list[Path]:
    root_dir = Path(root_dir)

    image_paths = [
        path
        for path in root_dir.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(image_paths)


class TripletPlaceDataset(Dataset):
    def __init__(
        self,
        anchor_dir: str | Path,
        database_dir: str | Path,
        transform: Callable | None = None,
        positive_tolerance: int = 3,
        negative_gap: int = 20,
    ):
        self.anchor_paths = find_images(anchor_dir)
        self.database_paths = find_images(database_dir)
        self.transform = transform or get_default_transform()
        self.positive_tolerance = positive_tolerance
        self.negative_gap = negative_gap

        if len(self.anchor_paths) == 0:
            raise ValueError(f"No anchor images found in {anchor_dir}")
        if len(self.database_paths) == 0:
            raise ValueError(f"No database images found in {database_dir}")

        self.database_indices = {
            path: image_index_from_path(path)
            for path in self.database_paths
        }

    def __len__(self) -> int:
        return len(self.anchor_paths)

    def __getitem__(self, index: int):
        anchor_path = self.anchor_paths[index]
        anchor_index = image_index_from_path(anchor_path)

        positive_candidates = [
            path
            for path in self.database_paths
            if abs(self.database_indices[path] - anchor_index) <= self.positive_tolerance
        ]

        negative_candidates = [
            path
            for path in self.database_paths
            if abs(self.database_indices[path] - anchor_index) >= self.negative_gap
        ]

        if len(positive_candidates) == 0:
            raise ValueError(f"No positive candidates found for {anchor_path}")
        if len(negative_candidates) == 0:
            raise ValueError(f"No negative candidates found for {anchor_path}")

        positive_path = random.choice(positive_candidates)
        negative_path = random.choice(negative_candidates)

        anchor_image = self._load_image(anchor_path)
        positive_image = self._load_image(positive_path)
        negative_image = self._load_image(negative_path)

        return {
            "anchor": anchor_image,
            "positive": positive_image,
            "negative": negative_image,
            "anchor_path": str(anchor_path),
            "positive_path": str(positive_path),
            "negative_path": str(negative_path),
        }

    def _load_image(self, path: Path):
        image = Image.open(path).convert("RGB")
        return self.transform(image)