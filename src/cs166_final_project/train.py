import json
from tqdm import tqdm
from pathlib import Path

import torch

from cs166_final_project.data.masks import MaskGenerator
from cs166_final_project.utils.batches import get_images
from cs166_final_project.utils.seed import set_seed


@torch.no_grad()
def _evaluate_loss(model, dataloader, mask_generator, device, mask_seed: int):
    model.eval()
    total_loss = 0.0
    generator = torch.Generator(device=device).manual_seed(mask_seed)

    for batch in dataloader:
        images = get_images(batch).to(device)
        masks = mask_generator(images, generator=generator)
        total_loss += model.compute_loss(images, masks).item()

    return total_loss / len(dataloader)


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    config,
    output_dir: str | Path,
):
    set_seed(config.train.seed)

    device = torch.device(config.train.device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    mask_generator = MaskGenerator(box_size=config.data.mask_size)

    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = []

    for epoch in range(config.train.epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_dataloader):
            images = get_images(batch).to(device)
            masks = mask_generator(images)

            loss = model.compute_loss(images, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        val_loss = _evaluate_loss(
            model,
            val_dataloader,
            mask_generator,
            device,
            config.eval.mask_seed,
        )

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"epoch {epoch + 1}/{config.train.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")

    torch.save(model.state_dict(), checkpoint_dir / "last.pt")
    with open(Path(output_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history

