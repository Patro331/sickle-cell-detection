"""
dataset.py
----------
Dataset loading and DataLoader creation for the sickle cell detection pipeline.

Datasets are expected to be organised in the following structure:
    data/processed/
        train/
            positive/
            negative/
        val/
            positive/
            negative/
        test/
            positive/
            negative/
"""

import os
from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


# ImageNet mean and std — used because EfficientNet and ResNet were pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224


def get_transforms(augment: bool = False) -> transforms.Compose:
    """
    Return a transform pipeline.

    Parameters
    ----------
    augment : bool
        If True, apply training-time augmentation.
        If False, apply resize and normalise only (for val/test).

    Returns
    -------
    transforms.Compose
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def get_datasets(processed_dir: str):
    """
    Load train, validation, and test ImageFolder datasets.

    Parameters
    ----------
    processed_dir : str
        Path to the processed data directory containing train/, val/, test/ subfolders.

    Returns
    -------
    tuple: (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = datasets.ImageFolder(
        os.path.join(processed_dir, "train"),
        transform=get_transforms(augment=True)
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(processed_dir, "val"),
        transform=get_transforms(augment=False)
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(processed_dir, "test"),
        transform=get_transforms(augment=False)
    )
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(processed_dir: str, batch_size: int = 32, num_workers: int = 2):
    """
    Create DataLoaders for train, validation, and test sets.

    A WeightedRandomSampler is applied to the training DataLoader to address
    class imbalance by oversampling the minority class during training.

    Parameters
    ----------
    processed_dir : str
        Path to the processed data directory.
    batch_size : int
        Number of images per batch. Default 32.
    num_workers : int
        Number of worker processes for data loading. Default 2.

    Returns
    -------
    tuple: (train_loader, val_loader, test_loader, class_to_idx)
    """
    train_dataset, val_dataset, test_dataset = get_datasets(processed_dir)

    # Compute per-class weights for WeightedRandomSampler
    class_counts  = Counter(train_dataset.targets)
    total         = sum(class_counts.values())
    class_weights = {cls: total / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_dataset.targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx
