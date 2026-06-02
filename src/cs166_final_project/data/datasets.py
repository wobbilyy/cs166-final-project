from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FFHQDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int = 32,
    ):
        self.root = Path(root)
        self.paths = sorted(self.root.glob("*.png"))

        if not self.paths:
            raise FileNotFoundError(
                f"no FFHQ images found in {self.root}. "
                "run src/.../preprocess_ffhq.py (takes like 8 hours...)"
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = Image.open(self.paths[idx]).convert("RGB")
        return {"image": self.transform(image)}

