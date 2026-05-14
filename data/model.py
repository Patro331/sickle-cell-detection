"""
model.py
--------
Model definitions for the sickle cell detection pipeline.

Three architectures are provided:
    1. BaselineCNN   — simple 3-block CNN trained from scratch (performance floor)
    2. EfficientNet  — EfficientNet-B0 fine-tuned with transfer learning
    3. ResNet        — ResNet-50 fine-tuned with transfer learning (best performer)
"""

import torch
import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    """
    Simple 3-block Convolutional Neural Network trained from scratch.

    Architecture:
        Block 1: Conv2d(3, 32)  -> BatchNorm -> ReLU -> MaxPool
        Block 2: Conv2d(32, 64) -> BatchNorm -> ReLU -> MaxPool
        Block 3: Conv2d(64, 128)-> BatchNorm -> ReLU -> MaxPool
        Classifier: Dropout(0.5) -> Linear(100352, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 1)

    Input size: 224 x 224 x 3
    Output: single logit for binary classification
    Total parameters: ~25.8M
    """

    def __init__(self):
        super(BaselineCNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def build_efficientnet(freeze_backbone: bool = False) -> nn.Module:
    """
    Build EfficientNet-B0 with a binary classification head.

    The original classifier head is replaced with:
        Dropout(0.3) -> Linear(in_features, 1)

    Parameters
    ----------
    freeze_backbone : bool
        If True, freeze all pretrained layers and train only the classifier head.
        If False (default), fine-tune all layers end-to-end.

    Returns
    -------
    nn.Module
        EfficientNet-B0 model ready for binary classification.

    Notes
    -----
    Total parameters: ~4.0M
    Pretrained on: ImageNet (IMAGENET1K_V1 weights)
    """
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1)
    )
    return model


def build_resnet(freeze_backbone: bool = False) -> nn.Module:
    """
    Build ResNet-50 with a binary classification head.

    The original fully connected layer is replaced with:
        Dropout(0.3) -> Linear(in_features, 1)

    Parameters
    ----------
    freeze_backbone : bool
        If True, freeze all pretrained layers and train only the classifier head.
        If False (default), fine-tune all layers end-to-end.

    Returns
    -------
    nn.Module
        ResNet-50 model ready for binary classification.

    Notes
    -----
    Total parameters: ~23.5M
    Pretrained on: ImageNet (IMAGENET1K_V1 weights)
    Best performing model — 98.57% accuracy, AUC-ROC 0.9973
    """
    model = models.resnet50(weights="IMAGENET1K_V1")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1)
    )
    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters in a model.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    dict with keys 'total' and 'trainable'
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
