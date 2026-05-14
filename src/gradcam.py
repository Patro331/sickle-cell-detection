"""
gradcam.py
----------
Gradient-weighted Class Activation Mapping (Grad-CAM) implementation
for the sickle cell detection pipeline.

Grad-CAM produces a heatmap highlighting the regions of an input image
that most influenced the model's prediction. It works by computing the
gradient of the predicted class score with respect to the feature maps
of the final convolutional layer, weighting those maps by their importance,
and applying ReLU to retain only positive contributions.

Reference:
    Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization. ICCV 2017.
"""

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt


class GradCAM:
    """
    Grad-CAM implementation using PyTorch forward and backward hooks.

    Parameters
    ----------
    model        : nn.Module — trained model
    target_layer : nn.Module — the convolutional layer to visualise
                   For ResNet-50 use model.layer4[-1]
                   For EfficientNet-B0 use model.features[-1]

    Example
    -------
    >>> gradcam = GradCAM(model, model.layer4[-1])
    >>> cam = gradcam.generate(input_tensor)
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given input.

        Parameters
        ----------
        input_tensor : torch.Tensor of shape (1, 3, H, W)
                       Must have requires_grad=True

        Returns
        -------
        np.ndarray of shape (H, W) — normalised heatmap in [0, 1]
        """
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward()

        gradients  = self.gradients.squeeze(0)
        activations = self.activations.squeeze(0)

        # Global average pool gradients over spatial dimensions
        weights = gradients.mean(dim=(1, 2))
        cam     = torch.zeros(activations.shape[1:], device=activations.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU — retain only positive activations
        cam = torch.relu(cam)

        # Normalise to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to match input image dimensions
        cam = cam.cpu().numpy()
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (w, h))

        return cam


def apply_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on an image.

    Parameters
    ----------
    image : np.ndarray of shape (H, W, 3) — normalised to [0, 1]
    cam   : np.ndarray of shape (H, W)    — normalised to [0, 1]
    alpha : float — heatmap opacity (default 0.4)

    Returns
    -------
    np.ndarray of shape (H, W, 3) — overlay image in [0, 1]
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap / 255.0
    overlay = alpha * heatmap + (1 - alpha) * image
    return np.clip(overlay, 0, 1)


def denormalise(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalisation for display purposes.

    Parameters
    ----------
    tensor : torch.Tensor of shape (3, H, W)

    Returns
    -------
    torch.Tensor of shape (3, H, W) with values clipped to [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def visualise_gradcam(
    model,
    gradcam: GradCAM,
    image_tensor: torch.Tensor,
    label: int,
    pred: int,
    prob: float,
    device: torch.device,
    save_path: str = None
):
    """
    Generate and display a three-panel Grad-CAM visualisation.

    Panels: Original Image | Grad-CAM Heatmap | Overlay

    Parameters
    ----------
    model        : nn.Module
    gradcam      : GradCAM instance
    image_tensor : torch.Tensor of shape (3, H, W)
    label        : int — true class label
    pred         : int — predicted class label
    prob         : float — predicted probability for positive class
    device       : torch.device
    save_path    : str or None — if provided, saves figure to this path
    """
    class_names = {0: "Negative", 1: "Positive"}

    input_tensor = image_tensor.unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    cam         = gradcam.generate(input_tensor)
    img_display = denormalise(image_tensor).permute(1, 2, 0).numpy()
    heatmap     = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap     = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay     = apply_heatmap(img_display, cam)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    title_col = "green" if label == pred else "red"
    fig.suptitle(
        f"True: {class_names[label]}  |  Pred: {class_names[pred]}  |  Conf: {prob:.3f}",
        fontsize=14, fontweight="bold", color=title_col
    )

    axes[0].imshow(img_display)
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap", fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()
