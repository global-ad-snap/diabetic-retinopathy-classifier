import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import random
from pathlib import Path
from io import BytesIO

import streamlit as st
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms.functional import resize, to_pil_image
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from src.models import DRClassifier
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import shap


# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DRClassifier().to(device)
model.load_state_dict(torch.load("src/best_model.pth", map_location=device))
model.eval()

transform = Compose([Resize((128, 128)), ToTensor()])
class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# === Sidebar Info ===
st.sidebar.title("Model Info")
st.sidebar.write("Architecture: Custom CNN")
st.sidebar.write("Input Size: 128×128")
st.sidebar.write("Trained on: APTOS 2019")

# Path to visuals directory for Grad-CAM samples
visuals_dir = Path(__file__).parent.parent / "visuals"

# Sidebar: Grad-CAM preview (optional)
if visuals_dir.exists():
    gradcam_images = list(visuals_dir.glob("gradcam_*.png"))

    if gradcam_images:
        chosen_image = random.choice(gradcam_images)
        st.sidebar.image(
            chosen_image,
            caption="Grad-CAM Sample",
            use_column_width=True
        )
    else:
        st.sidebar.info("Grad-CAM visualizations will appear after prediction.")
else:
    st.sidebar.info("Grad-CAM visualizations generated at runtime.")

# === Streamlit UI ===
st.title("Diabetic Retinopathy Classifier")
st.write("Upload a retinal image to get a prediction and visual explanation.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Prepare input tensor
    input_tensor = transform(image).unsqueeze(0).to(device)

    # === Run model and get prediction ===
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    pred_class = torch.argmax(output, dim=1).item()

    # === Grad-CAM Overlay ===
    st.write("### Grad-CAM Overlay")
    
    # Initialize GradCAM extractor
    cam_extractor = GradCAM(model, target_layer=model.conv[3])

    # Run the image through the model to populate CAM extractor
    output = model(input_tensor)
    pred_class = torch.argmax(output, dim=1).item()

    # Generate activation map
    activation_map = cam_extractor(pred_class, output)

    # Convert input tensor back to PIL
    original = to_pil_image(input_tensor.squeeze().cpu())

    # Resize activation map to original image size
    activation_map_resized = resize(
        activation_map[0].unsqueeze(0),
        size=original.size[::-1]
    )[0]

    # Convert to heatmap and overlay
    heatmap = to_pil_image(activation_map_resized.cpu(), mode='F')
    overlay = overlay_mask(original, heatmap, alpha=0.5)

    st.image(overlay, caption="Grad-CAM", use_container_width=True)

    # Download button for Grad-CAM
    buf = BytesIO()
    overlay.save(buf, format="PNG")
    st.download_button("Download Grad-CAM", buf.getvalue(), file_name="gradcam.png")

    # === Prediction ===
    st.write(f"**Predicted Class:** {class_names[pred_class]}")

    # === Clean up TorchCAM hooks before SHAP ===
    try:
        # Remove TorchCAM hooks
        if hasattr(cam_extractor, '_hooks'):
            cam_extractor._hooks.clear()
        if hasattr(cam_extractor, 'hook_handles'):
            for h in cam_extractor.hook_handles:
                h.remove()
            cam_extractor.hook_handles.clear()
        
        # Delete the extractor
        del cam_extractor
        
        # Clear any remaining hooks on the model
        model._forward_hooks.clear()
        model._forward_pre_hooks.clear()
        model._backward_hooks.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        st.write("✅ Cleared all TorchCAM hooks.")
    except Exception as e:
        st.write(f"⚠️ Hook cleanup warning: {str(e)}")

    # === SHAP Explanation ===
    st.write("### SHAP Explanation")

    # Initialize variable for comparison section
    buf_img = None

    try:
        # Prepare input with gradients enabled
        input_tensor_shap = transform(image).unsqueeze(0).to(device)
        input_tensor_shap.requires_grad_(True)

        st.write(f"Input tensor shape for SHAP: {input_tensor_shap.shape}")

        # Create background baselines
        background_list = []
        
        # 1. Heavy blur baseline
        blurred_heavy = image.filter(ImageFilter.GaussianBlur(radius=20))
        background_list.append(transform(blurred_heavy))
        
        # 2. Black baseline
        background_list.append(torch.zeros((3, 128, 128)))
        
        # 3. White baseline  
        background_list.append(torch.ones((3, 128, 128)))
        
        background = torch.stack(background_list).to(device)
        st.write(f"Background shape for SHAP: {background.shape}")

        # Use GradientExplainer
        with torch.enable_grad():
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(input_tensor_shap)

        # Validate SHAP output
        if shap_values is None or (isinstance(shap_values, list) and len(shap_values) == 0):
            st.error("❌ SHAP failed to generate values.")
            st.stop()

        # Extract SHAP values for predicted class
        if isinstance(shap_values, list):
            sv = shap_values[pred_class]
        else:
            sv = shap_values
        
        # Convert to numpy if needed
        if torch.is_tensor(sv):
            sv = sv.detach().cpu().numpy()
        
        st.write(f"SHAP values shape: {sv.shape}")
        
        # Handle different tensor shapes
        if sv.ndim == 5:  # [B, C, H, W, N] - batch, channels, height, width, background
            sv = sv[0]  # Remove batch dimension -> [C, H, W, N]
            sv = np.mean(sv, axis=-1)  # Average over background -> [C, H, W]
        elif sv.ndim == 4:  # [B, C, H, W] or [C, H, W, N]
            if sv.shape[0] == 1:  # [1, C, H, W]
                sv = sv[0]  # -> [C, H, W]
            elif sv.shape[-1] <= 5:  # [C, H, W, N] (background samples)
                sv = np.mean(sv, axis=-1)  # -> [C, H, W]
            else:
                sv = sv[0]  # Take first sample
        
        # Take absolute values and average across color channels
        sv = np.abs(sv)
        if sv.ndim == 3:  # [C, H, W]
            sv = np.mean(sv, axis=0)  # -> [H, W]
        
        st.write(f"Final SHAP map shape: {sv.shape}")
        
        # Apply Gaussian smoothing to reduce grid artifacts
        sv = gaussian_filter(sv, sigma=1.5)
        
        # Normalize with percentile clipping
        sv_flat = sv.flatten()
        low, high = np.percentile(sv_flat[sv_flat > 0], [5, 95])
        sv = np.clip(sv, low, high)
        sv = (sv - sv.min()) / (sv.max() - sv.min() + 1e-8)
        
        # Apply gamma correction for better visibility
        sv = np.power(sv, 0.5)

        # Prepare original image for overlay
        img_np = input_tensor_shap.squeeze().detach().cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        # Create overlay visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_np)
        
        # Overlay SHAP heatmap
        im = ax.imshow(sv, cmap="hot", alpha=0.6, vmin=0, vmax=1, interpolation='bilinear')
        ax.axis("off")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attribution Intensity', rotation=270, labelpad=15)

        # Save to buffer
        buf_img = BytesIO()
        fig.savefig(buf_img, format="png", bbox_inches="tight", pad_inches=0.05, dpi=150)
        buf_img.seek(0)
        plt.close(fig)

        st.image(buf_img, caption="SHAP Explanation", use_container_width=True)
        st.download_button("Download SHAP Explanation", buf_img.getvalue(), file_name="shap_explanation.png")

    except Exception as e:
        st.error(f"❌ SHAP Explanation failed: {str(e)}")
        st.write("This can happen if the model architecture is not fully compatible with SHAP's gradient methods.")
        import traceback
        st.code(traceback.format_exc())
        
    # === Compare Grad-CAM and SHAP ===
    st.write("### Compare Grad-CAM and SHAP")
    compare = st.checkbox("Show overlays side-by-side")

    if compare:
        col1, col2 = st.columns(2)
        with col1:
            st.image(overlay, caption="Grad-CAM", use_container_width=True)
        with col2:
            if buf_img is not None:
                buf_img.seek(0)
                st.image(buf_img, caption="SHAP Explanation", use_container_width=True)
            else:
                st.warning("SHAP explanation not available")