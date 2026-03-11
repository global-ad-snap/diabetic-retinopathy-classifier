# src/ app.py

import sys
import warnings
warnings.filterwarnings("ignore", message=".*use_column_width.*")
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
from pathlib import Path
from io import BytesIO

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torchvision import transforms, models
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import shap

# ═══════════════════════════════════════════════════════════════════════════════
# Model Definition — must match train_fixed.py exactly
# ═══════════════════════════════════════════════════════════════════════════════
class DRClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone   = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


# ═══════════════════════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    m = DRClassifier().to(device)
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.eval()
    return m

model = load_model()

# ResNet expects 224×224 with ImageNet normalisation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("Model Info")
st.sidebar.write("**Architecture:** ResNet-18 (pretrained)")
st.sidebar.write("**Input Size:** 224×224")
st.sidebar.write("**Trained on:** APTOS 2019")
st.sidebar.markdown("---")
st.sidebar.markdown("**Validation Metrics**")
st.sidebar.write("ROC-AUC : 0.9408")
st.sidebar.write("F1 Score : 0.7842")
st.sidebar.write("Kappa   : 0.6738")
st.sidebar.write("Accuracy : 77.8%")

visuals_dir = Path(__file__).parent.parent / "visuals"
if visuals_dir.exists():
    gradcam_images = list(visuals_dir.glob("gradcam_*.png"))
    if gradcam_images:
        st.sidebar.image(
            random.choice(gradcam_images),
            caption="Grad-CAM Sample",
            use_container_width=True
        )

# ═══════════════════════════════════════════════════════════════════════════════
# Main UI
# ═══════════════════════════════════════════════════════════════════════════════
st.title("Diabetic Retinopathy Classifier")
st.write("Upload a retinal fundus image to get a severity prediction and visual explanation.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Prepare tensor
    input_tensor = transform(image).unsqueeze(0).to(device)

    # ── Prediction ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        output = model(input_tensor)
        probs  = torch.softmax(output, dim=1)[0]
    pred_class = torch.argmax(probs).item()

    st.markdown(f"### Predicted: **{class_names[pred_class]}** (Stage {pred_class})")

    # Probability bar chart
    fig_prob, ax_prob = plt.subplots(figsize=(7, 3))
    colors = ['#2ecc71' if i == pred_class else '#95a5a6' for i in range(5)]
    ax_prob.barh(class_names, probs.cpu().numpy(), color=colors)
    ax_prob.set_xlabel("Probability")
    ax_prob.set_title("Class Probabilities")
    ax_prob.set_xlim(0, 1)
    for i, v in enumerate(probs.cpu().numpy()):
        ax_prob.text(v + 0.01, i, f"{v:.2%}", va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_prob)
    plt.close(fig_prob)

    # ── Grad-CAM ───────────────────────────────────────────────────────────────
    st.write("### Grad-CAM Overlay")
    st.write("Highlights retinal regions that influenced the prediction.")

    try:
        # Target the last conv layer inside ResNet backbone
        # ResNet children: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool]
        # Index 7 = layer4 (last conv block) — best layer for Grad-CAM
        target_layer = list(model.backbone.children())[7]
        cam_extractor = GradCAM(model, target_layer=target_layer)

        # Need gradients for CAM
        output_cam = model(input_tensor)
        pred_for_cam = torch.argmax(output_cam, dim=1).item()
        activation_map = cam_extractor(pred_for_cam, output_cam)

        # Convert to overlay
        original_pil = transforms.ToPILImage()(input_tensor.squeeze().cpu())
        heatmap_pil  = to_pil_image(activation_map[0].squeeze().cpu(), mode='F')
        overlay      = overlay_mask(original_pil, heatmap_pil, alpha=0.5)

        st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=True)

        buf_cam = BytesIO()
        overlay.save(buf_cam, format="PNG")
        st.download_button("Download Grad-CAM", buf_cam.getvalue(), file_name="gradcam.png")

        # Clean up hooks
        try:
            if hasattr(cam_extractor, 'hook_handles'):
                for h in cam_extractor.hook_handles:
                    h.remove()
            del cam_extractor
            model._forward_hooks.clear()
            model._forward_pre_hooks.clear()
            model._backward_hooks.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    except Exception as e:
        st.error(f"Grad-CAM failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

    # ── SHAP ───────────────────────────────────────────────────────────────────
    st.write("### SHAP Explanation")
    st.write("Pixel-level attribution showing what drove this prediction.")

    buf_img = None
    try:
        input_tensor_shap = transform(image).unsqueeze(0).to(device)
        input_tensor_shap.requires_grad_(True)

        # Background baselines at 224×224 to match ResNet input
        blurred = image.filter(ImageFilter.GaussianBlur(radius=20))
        background = torch.stack([
            transform(blurred),                   # blurred baseline
            torch.zeros(3, 224, 224),             # black baseline
            torch.ones(3, 224, 224)               # white baseline
        ]).to(device)

        with torch.enable_grad():
            explainer   = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(input_tensor_shap)

        # Extract values for predicted class
        sv = shap_values[pred_class] if isinstance(shap_values, list) else shap_values
        if torch.is_tensor(sv):
            sv = sv.detach().cpu().numpy()

        # Reduce to [H, W]
        if sv.ndim == 5:
            sv = np.mean(sv[0], axis=-1)      # [C, H, W, N] → [C, H, W]
        elif sv.ndim == 4:
            sv = sv[0] if sv.shape[0] == 1 else np.mean(sv, axis=-1)
        sv = np.abs(sv)
        if sv.ndim == 3:
            sv = np.mean(sv, axis=0)           # [C, H, W] → [H, W]

        # Smooth + normalise
        sv = gaussian_filter(sv, sigma=1.5)
        sv_flat = sv.flatten()
        pos = sv_flat[sv_flat > 0]
        if len(pos) > 0:
            low, high = np.percentile(pos, [5, 95])
            sv = np.clip(sv, low, high)
        sv = (sv - sv.min()) / (sv.max() - sv.min() + 1e-8)
        sv = np.power(sv, 0.5)  # gamma correction

        # Denormalise image for display
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_np = input_tensor_shap.squeeze().detach().cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * std + mean, 0, 1)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_np)
        im   = ax.imshow(sv, cmap="hot", alpha=0.6, vmin=0, vmax=1, interpolation='bilinear')
        ax.axis("off")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attribution Intensity', rotation=270, labelpad=15)

        buf_img = BytesIO()
        fig.savefig(buf_img, format="png", bbox_inches="tight", dpi=150)
        buf_img.seek(0)
        plt.close(fig)

        st.image(buf_img, caption="SHAP Explanation", use_container_width=True)
        st.download_button("Download SHAP", buf_img.getvalue(), file_name="shap_explanation.png")

    except Exception as e:
        st.error(f"SHAP failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

    # ── Side-by-side comparison ────────────────────────────────────────────────
    st.write("### Compare Grad-CAM and SHAP")
    if st.checkbox("Show overlays side-by-side"):
        col1, col2 = st.columns(2)
        with col1:
            st.image(overlay, caption="Grad-CAM", use_container_width=True)
        with col2:
            if buf_img is not None:
                buf_img.seek(0)
                st.image(buf_img, caption="SHAP", use_container_width=True)
            else:
                st.warning("SHAP not available") 
