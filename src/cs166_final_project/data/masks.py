import torch


class MaskGenerator:

    def __init__(self, box_size: int = 14):
        self.box_size = box_size

    def _generate_single_mask(
        self,
        h: int,
        w: int,
        device = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        box_size = min(self.box_size, h, w)

        mask = torch.ones((1, h, w), dtype=torch.float32, device=device)

        y = torch.randint(
            0,
            h - box_size + 1,
            (1,),
            device=device,
            generator=generator,
        ).item()
        x = torch.randint(
            0,
            w - box_size + 1,
            (1,),
            device=device,
            generator=generator,
        ).item()

        mask[:, y:y + box_size, x: x + box_size] = 0.0

        return mask

    def __call__(
        self,
        img_tensor: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if img_tensor.dim() == 3:
            return self._generate_single_mask(
                img_tensor.shape[1],
                img_tensor.shape[2],
                device=img_tensor.device,
                generator=generator,
            )

        if img_tensor.dim() == 4:
            return torch.stack([
                self._generate_single_mask(
                    img_tensor.shape[2],
                    img_tensor.shape[3],
                    device=img_tensor.device,
                    generator=generator,
                )
                for _ in range(img_tensor.shape[0])
            ])

        raise ValueError(
            "image data is of wrong dimension."
        )

