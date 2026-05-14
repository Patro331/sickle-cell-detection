
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Device ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load Model ──────────────────────────────────────────────
def load_model():
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1)
    )
    weights_path = hf_hub_download(
        repo_id="Patrovas/sickle-cell-resnet50",
        filename="resnet50_unfrozen_best.pt"
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# ── Transforms ──────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Grad-CAM ────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward()
        gradients = self.gradients.squeeze(0)
        activations = self.activations.squeeze(0)
        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        return cam

gradcam = GradCAM(model, model.layer4[-1])

# ── Predict Function ────────────────────────────────────────
def predict(image):
    if image is None:
        return "Please upload an image.", None

    img = Image.fromarray(image).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0

    input_tensor = transform(img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    cam = gradcam.generate(input_tensor)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    pred_label = "SICKLE CELL DETECTED" if prob >= 0.5 else "NORMAL"
    confidence = prob if prob >= 0.5 else 1 - prob
    color = "🔴" if prob >= 0.5 else "🟢"

    result_text = (
        f"{color} Prediction: {pred_label}\n"
        f"Confidence: {confidence*100:.1f}%\n\n"
    )

    if prob >= 0.5:
        result_text += (
            "⚠️ Sickle cells detected. Please refer this patient "
            "to a trained clinician for confirmation and further care."
        )
    else:
        result_text += (
            "✅ No sickle cells detected. Continue routine monitoring."
        )

    result_text += (
        "\n\n─────────────────────────────\n"
        "⚕️ DISCLAIMER: This tool is a screening aid only and does "
        "not replace clinical diagnosis by a trained medical professional."
    )

    # Grad-CAM overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.4 * heatmap + 0.6 * img_array
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Prediction: {pred_label} ({confidence*100:.1f}% confidence)",
        fontsize=14, fontweight="bold",
        color="red" if prob >= 0.5 else "green"
    )

    axes[0].imshow(img_array)
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap", fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()

    gradcam_path = "/tmp/gradcam_output.png"
    plt.savefig(gradcam_path, dpi=150, bbox_inches="tight")
    plt.close()

    return result_text, gradcam_path


# ── Gradio Interface ─────────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Blood Smear Image", type="numpy"),
    outputs=[
        gr.Textbox(label="Prediction Result", lines=8),
        gr.Image(label="Grad-CAM Explainability")
    ],
    title="🔬 Sickle Cell Detection — Point-of-Care Screening Tool",
    description=(
        "Upload a blood smear image captured through a microscope. "
        "The model will predict whether sickle cells are present and "
        "highlight the regions it focused on using Grad-CAM.\n\n"
        "**Dataset:** Tushabe et al. (2024) Ugandan Clinical Microscopy Dataset\n"
        "**Model:** ResNet-50 with Transfer Learning — 98.57% accuracy, "
        "AUC-ROC 0.9973\n"
        "**Developed by:** Okidi Patrovas Gaabriel | Makerere University | "
        "MSB7216: Deep Learning for Health Data"
    ),
    examples=[],
    theme=gr.themes.Soft(),
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()
