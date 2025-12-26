# src/explain.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import shap

def generate_gradcam(model, val_df, base_img_dir, output_dir, target_layer, device, num_samples=5):
    os.makedirs(output_dir, exist_ok=True)
    transform = Compose([Resize((128, 128)), ToTensor()])
    cam_extractor = GradCAM(model, target_layer=target_layer)
    model.eval()

    for i in range(num_samples):
        sample_id = val_df.iloc[i]['id_code']
        sample_path = os.path.join(base_img_dir, f"{sample_id}.png")
        if not os.path.exists(sample_path):
            print(f"❌ Missing image: {sample_path}")
            continue

        image = Image.open(sample_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        activation_map = cam_extractor(pred_class, output)
        original = to_pil_image(input_tensor.squeeze().cpu())
        heatmap = to_pil_image(activation_map[0].cpu(), mode='F')
        overlay = overlay_mask(original, heatmap, alpha=0.5)

        save_path = os.path.join(output_dir, f"gradcam_{sample_id}.png")
        overlay.save(save_path)
        print(f"✅ Grad-CAM saved: {save_path}")

def generate_shap(model, val_df, base_img_dir, output_dir, device, num_samples=5):
    os.makedirs(output_dir, exist_ok=True)
    transform = Compose([Resize((128, 128)), ToTensor()])
    sample_images = []
    sample_ids = []

    for i in range(num_samples):
        sample_id = val_df.iloc[i]['id_code']
        sample_path = os.path.join(base_img_dir, f"{sample_id}.png")
        if os.path.exists(sample_path):
            img = Image.open(sample_path).convert('RGB')
            tensor = transform(img).unsqueeze(0)
            sample_images.append(tensor)
            sample_ids.append(sample_id)

    input_batch = torch.cat(sample_images).to(device)
    input_batch.requires_grad_()

    class SHAPWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)

    wrapped_model = SHAPWrapper(model)
    model.eval()

    explainer = shap.GradientExplainer(wrapped_model, input_batch)
    shap_values = explainer.shap_values(input_batch)

    for i in range(len(shap_values[0])):
        img_np = input_batch[i].detach().cpu().squeeze().permute(1, 2, 0).numpy()
        shap_img = shap_values[0][i].transpose(1, 2, 0)

        plt.figure(figsize=(4, 4))
        shap.image_plot([shap_img], img_np, show=False)
        plt.savefig(os.path.join(output_dir, f"shap_{sample_ids[i]}.png"))
        plt.close()
        print(f"✅ SHAP saved: shap_{sample_ids[i]}.png")

