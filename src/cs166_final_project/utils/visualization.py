"""
made with AI
"""

from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from cs166_final_project.data.masks import MaskGenerator
from cs166_final_project.utils.batches import get_images


def grid(images: torch.Tensor, nrow: int, normalized: bool = True) -> Image.Image:
    images = images.detach().cpu()

    if normalized:
        images = ((images + 1.0) / 2.0).clamp(0.0, 1.0)
    else:
        images = images.clamp(0.0, 1.0)

    grid = make_grid(images, nrow=nrow, padding=2, pad_value=1.0)
    return to_pil_image(grid)


def save_labeled_rows(
    rows: list[tuple[str, torch.Tensor, bool]],
    output_path: str | Path,
    nrow: int,
) -> None:
    label_width = 100
    row_images = [(label, grid(images, nrow=nrow, normalized=normalized))
                  for label, images, normalized in rows]
    width = label_width + max(image.width for _, image in row_images)
    height = sum(image.height for _, image in row_images)

    canvas = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(canvas)

    y = 0
    for label, image in row_images:
        draw.text((10, y + image.height // 2 - 6), label, fill="black")
        canvas.paste(image.convert("RGB"), (label_width, y))
        y += image.height

    canvas.save(output_path)


def save_data_examples(
    dataloader,
    output_path: str | Path,
    mask_size: int,
    mask_seed: int = 0,
    max_images: int = 16,
) -> None:
    batch = next(iter(dataloader))
    images = get_images(batch)
    images = images[:max_images]

    generator = torch.Generator(device=images.device).manual_seed(mask_seed)
    masks = MaskGenerator(box_size=mask_size)(images, generator=generator)
    masked_images = images * masks

    save_labeled_rows(
        [
            ("Original", images, True),
            ("Masked", masked_images, True),
        ],
        output_path,
        nrow=len(images),
    )


@torch.no_grad()
def save_inpainting_examples(
    model,
    dataloader,
    output_path: str | Path,
    mask_size: int,
    mask_seed: int,
    device,
    max_images: int = 16,
) -> None:
    batch = next(iter(dataloader))
    images = get_images(batch)
    images = images[:max_images].to(device)

    generator = torch.Generator(device=device).manual_seed(mask_seed)
    masks = MaskGenerator(box_size=mask_size)(images, generator=generator)
    masked_images = images * masks
    inpainted = model.inpaint(images, masks)
    error = ((inpainted - images).abs() * (1.0 - masks) / 2.0).clamp(0.0, 1.0)

    save_labeled_rows(
        [
            ("Original", images, True),
            ("Masked", masked_images, True),
            ("Inpainted", inpainted, True),
            ("Abs error", error, False),
        ],
        output_path,
        nrow=len(images),
    )
