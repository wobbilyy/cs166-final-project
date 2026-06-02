import json
import math
from tqdm import tqdm
from pathlib import Path

import torch

from cs166_final_project.data.masks import MaskGenerator
from cs166_final_project.utils.batches import get_images
from cs166_final_project.utils.visualization import save_inpainting_examples


def _count_parameters(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _network_evaluations_per_sample(model) -> int:
    if hasattr(model, "num_timesteps"):
        return model.num_timesteps

    if hasattr(model, "num_steps"):
        return model.num_steps

    return 1


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    config,
    output_dir: str | Path,
):
    device = torch.device(config.train.device)
    model = model.to(device)
    model.eval()

    mask_generator = MaskGenerator(box_size=config.data.mask_size)
    generator = torch.Generator(device=device).manual_seed(config.eval.mask_seed)
    total_mse = 0.0
    total_mae = 0.0

    for batch in tqdm(dataloader):
        images = get_images(batch).to(device)
        masks = mask_generator(images, generator=generator)
        inpainted = model.inpaint(images, masks)

        missing = 1.0 - masks

        denom = (missing.sum() * images.shape[1]).clamp_min(1.0)
        total_mse += (((inpainted - images) ** 2) * missing).sum().item() / denom.item()
        total_mae += ((inpainted - images).abs() * missing).sum().item() / denom.item()

    mse = total_mse / len(dataloader)
    mae = total_mae / len(dataloader)
    psnr = 20.0 * math.log10(2.0) - 10.0 * math.log10(max(mse, 1e-12))

    metrics = {
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
        "model": config.model.name,
        "dataset": config.data.name,
        "mask_size": config.data.mask_size,
        "mask_seed": config.eval.mask_seed,
        "num_parameters": _count_parameters(model),
        "network_evaluations_per_sample": _network_evaluations_per_sample(model),
    }
    print(f"eval mse={mse:.4f} mae={mae:.4f} psnr={psnr:.2f}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if config.eval.save_plots:
        save_inpainting_examples(
            model,
            dataloader,
            output_dir / "inpainting_examples.png",
            mask_size=config.data.mask_size,
            mask_seed=config.eval.mask_seed,
            device=device,
            max_images=config.eval.max_images,
        )

    return metrics

