"""
Command-line inference: run a trained model on a single chest X-ray image,
print the prediction + confidence, and save a Grad-CAM visualization.

Usage:
    python predict.py --image path/to/xray.jpg --weights model/pneumonia_model.pt
"""

import argparse
import os

from PIL import Image

from utils.model_def import CLASS_NAMES, get_device, load_model, find_target_layer
from utils.preprocess import load_and_preprocess, unnormalize
from utils.gradcam import GradCAM, overlay_heatmap
from utils.visualization import plot_prediction_triptych


def predict(image_path: str, weights_path: str, backbone: str = "resnet50", out_dir: str = "outputs"):
    device = get_device()
    model = load_model(weights_path, backbone_name=backbone, device=device)
    target_layer = find_target_layer(model)

    img_tensor = load_and_preprocess(image_path, device)[0]  # drop batch dim -> [C,H,W]

    cam = GradCAM(model, target_layer)
    heatmap, pred_idx, confidence = cam(img_tensor)
    cam.remove_hooks()

    pred_label = CLASS_NAMES[pred_idx]
    original_np = unnormalize(img_tensor)
    heatmap_resized, _, overlay = overlay_heatmap(original_np, heatmap)

    print(f"Image:      {image_path}")
    print(f"Prediction: {pred_label}")
    print(f"Confidence: {confidence * 100:.2f}%")

    os.makedirs(out_dir, exist_ok=True)
    fig = plot_prediction_triptych(original_np, heatmap_resized, overlay,
                                    pred_label=pred_label, confidence=confidence)
    out_path = os.path.join(out_dir, f"gradcam_{os.path.splitext(os.path.basename(image_path))[0]}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved visualization to: {out_path}")

    return pred_label, confidence, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pneumonia detection on a single chest X-ray.")
    parser.add_argument("--image", required=True, help="Path to a chest X-ray image (jpg/png).")
    parser.add_argument("--weights", default="model/pneumonia_model.pt", help="Path to trained model weights.")
    parser.add_argument("--backbone", default="resnet50", help="timm backbone name used to train the weights.")
    parser.add_argument("--out_dir", default="outputs", help="Directory to save the Grad-CAM figure.")
    args = parser.parse_args()

    predict(args.image, args.weights, args.backbone, args.out_dir)
