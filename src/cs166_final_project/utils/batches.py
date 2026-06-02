

def get_images(batch):
    if isinstance(batch, dict):
        return batch["image"]

    if isinstance(batch, (tuple, list)):
        return batch[0]

    return batch

