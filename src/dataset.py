from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str | Path, transform: Callable | None = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = self._find_images(self.root_dir)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, str(image_path)

    @staticmethod
    def _find_images(root_dir: Path) -> list[Path]:
        image_paths = [
            path
            for path in root_dir.rglob("*")
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return sorted(image_paths)


def get_default_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )