import os
from typing import Tuple, Optional, Callable

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.io import read_image as tv_read_image


def _build_transform(train: bool):
    ops = []
    # Force grayscale (1 channel) to match the CNN input
    ops.append(v2.Grayscale(num_output_channels=1))
    if train:
        ops.append(v2.RandomHorizontalFlip(p=0.5))
    # float32 in [0,1]
    ops.append(v2.ToDtype(dtype=torch.float32, scale=True))
    return v2.Compose(ops)


def make_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    loader: Optional[Callable] = None,  # custom read_image if you have one
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Expects:
      data_dir/
        train/
          NORMAL/, PNEUMONIA/, ...
        test/
          NORMAL/, PNEUMONIA/, ...
    We’ll use test/ as validation in training; eval.py will also use test/.
    """
    if loader is None:
        loader = tv_read_image

    train_tf = _build_transform(train=True)
    eval_tf = _build_transform(train=False)

    # cast labels to float and keep shape [1] to match BCELoss with sigmoid output
    to_float_target = lambda x: torch.tensor([float(x)], dtype=torch.float32)

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_ds = datasets.ImageFolder(
        train_dir, transform=train_tf, target_transform=to_float_target, loader=loader
    )
    val_ds = datasets.ImageFolder(
        test_dir, transform=eval_tf, target_transform=to_float_target, loader=loader
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # No separate test loader here (eval.py will open test/ again)
    return train_loader, val_loader, None
