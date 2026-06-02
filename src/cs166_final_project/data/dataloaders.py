from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

from cs166_final_project.data.datasets import FFHQDataset


def _transform(image_size: int, grayscale: bool = False):
    mean = (0.5,) if grayscale else (0.5, 0.5, 0.5)
    std = (0.5,) if grayscale else (0.5, 0.5, 0.5)

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _split(dataset, val_split: float):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    # make sure alwys the same..
    generator = torch.Generator().manual_seed(0)
    return random_split(dataset, [train_size, val_size], generator=generator)


def get_mnist_dataloaders(
    data_dir: Path,
    batch_size: int,
    image_size: int = 28,
    num_workers: int = 4,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    transform = _transform(image_size=image_size, grayscale=True)
    data_dir = Path(data_dir) / "mnist"

    train = MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test = MNIST(root=str(data_dir), train=False, download=True, transform=transform)

    train_dataset, val_dataset = _split(train, val_split=val_split)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def get_cifar_dataloaders(
    data_dir: Path,
    batch_size: int,
    image_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    transform = _transform(image_size=image_size, grayscale=False)
    data_dir = Path(data_dir) / "cifar"

    train = CIFAR10(root=str(data_dir), train=True, download=True, transform=transform)
    test = CIFAR10(root=str(data_dir), train=False, download=True, transform=transform)

    train_dataset, val_dataset = _split(train, val_split=val_split)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def get_ffhq_dataloaders(
    data_dir: Path,
    batch_size: int,
    image_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.1,
    test_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    dataset = FFHQDataset(
        Path(data_dir) / "ffhq" / f"{image_size}x{image_size}",
        image_size=image_size,
    )

    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    generator = torch.Generator().manual_seed(0)

    train, test = random_split(
        dataset, [train_size, test_size], generator=generator
    )

    train_dataset, val_dataset = _split(train, val_split=val_split)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

