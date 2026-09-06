from __future__ import annotations

from torchvision import transforms


def build_transforms(config: dict, train: bool):
    data_cfg = config["data"]
    size = int(data_cfg["image_size"])
    grayscale = bool(data_cfg.get("grayscale", True))
    channels = 1 if grayscale else 3
    operations = [transforms.Grayscale(num_output_channels=channels), transforms.Resize((size, size))]
    if train:
        aug = data_cfg.get("augmentation", {})
        operations.extend(
            [
                transforms.RandomHorizontalFlip(float(aug.get("horizontal_flip", 0.0))),
                transforms.RandomRotation(float(aug.get("rotation_degrees", 0.0))),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(float(aug.get("affine_translate", 0.0)),) * 2,
                    scale=tuple(aug.get("affine_scale", [1.0, 1.0])),
                ),
                transforms.ColorJitter(
                    brightness=float(aug.get("brightness", 0.0)),
                    contrast=float(aug.get("contrast", 0.0)),
                ),
            ]
        )
    operations.append(transforms.ToTensor())
    normalization = data_cfg.get("normalization", {})
    default_mean = [0.5] * channels
    default_std = [0.5] * channels
    mean = list(normalization.get("mean", default_mean))
    std = list(normalization.get("std", default_std))
    if len(mean) != channels or len(std) != channels:
        raise ValueError(f"Normalization requires {channels} mean/std values; received mean={mean}, std={std}")
    operations.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(operations)
