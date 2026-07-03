"""
Streamlit app for AI-Assisted Pneumonia Detection.

Run with:
    streamlit run app.py
"""

import os
import io

import streamlit as st
from PIL import Image

from utils.model_def import CLASS_NAMES, get_device, load_model, find_target_layer
from utils.preprocess import load_and_preprocess, unnormalize
from utils.gradcam import GradCAM, overlay_heatmap
from utils.visualization import plot_confidence_histogram, plot_prediction_distribution
from utils.report import generate_pdf_report, DISCLAIMER

st.set_page_config(page_title="AI Pneumonia Detection", page_icon="🫁", layout="wide")

MODEL_PATH = "model/pneumonia_model.pt"
BACKBONE = "resnet50"


@st.cache_resource(show_spinner="Loading model...")
def get_cached_model(weights_path: str, backbone: str):
    device = get_device()
    model = load_model(weights_path, backbone_name=backbone, device=device)
    target_layer = find_target_layer(model)
    return model, target_layer, device


def init_session_state():
    if "history" not in st.session_state:
        # list of dicts: {label, confidence}
        st.session_state.history = []


def run_inference(pil_image: Image.Image, model, target_layer, device):
    img_tensor = load_and_preprocess(pil_image, device)[0]  # [C,H,W]
    cam = GradCAM(model, target_layer)
    heatmap, pred_idx, confidence = cam(img_tensor)
    cam.remove_hooks()

    original_np = unnormalize(img_tensor)
    heatmap_resized, _, overlay = overlay_heatmap(original_np, heatmap)
    return CLASS_NAMES[pred_idx], confidence, original_np, overlay


def main():
    init_session_state()

    st.title("🫁 AI-Assisted Pneumonia Detection")
    st.caption("Deep learning + explainable AI (Grad-CAM) for chest X-ray triage support.")

    with st.sidebar:
        st.header("About")
        st.write(
            "This app classifies chest X-rays into **Normal**, **Bacterial Pneumonia**, "
            "or **Viral Pneumonia** using a transfer-learned CNN (ResNet50 backbone), "
            "and visualizes the regions driving the prediction using Grad-CAM."
        )
        st.warning(DISCLAIMER)

        if not os.path.exists(MODEL_PATH):
            st.error(
                f"No trained model found at `{MODEL_PATH}`.\n\n"
                "Run `python train.py` first, or place your trained weights there."
            )

    tab_predict, tab_dashboard = st.tabs(["🔍 Predict", "📊 Dashboard"])

    with tab_predict:
        uploaded = st.file_uploader("Upload a chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

        if uploaded is not None:
            if not os.path.exists(MODEL_PATH):
                st.stop()

            pil_image = Image.open(uploaded).convert("RGB")
            model, target_layer, device = get_cached_model(MODEL_PATH, BACKBONE)

            with st.spinner("Running inference..."):
                pred_label, confidence, original_np, overlay = run_inference(
                    pil_image, model, target_layer, device
                )

            st.session_state.history.append({"label": pred_label, "confidence": confidence})

            col1, col2 = st.columns(2)
            with col1:
                st.image(original_np, caption="Original X-Ray", use_container_width=True)
            with col2:
                st.image(overlay, caption="Grad-CAM Overlay (model focus region)", use_container_width=True)

            badge_color = "green" if pred_label == "Normal" else "red"
            st.markdown(f"### Prediction: :{badge_color}[{pred_label}]")
            st.progress(min(max(confidence, 0.0), 1.0))
            st.write(f"**Confidence:** {confidence * 100:.1f}%")

            st.info(DISCLAIMER)

            overlay_pil = Image.fromarray((overlay * 255).astype("uint8"))
            pdf_bytes = generate_pdf_report(pil_image, overlay_pil, pred_label, confidence)
            st.download_button(
                "⬇️ Download PDF report",
                data=pdf_bytes,
                file_name="pneumonia_report.pdf",
                mime="application/pdf",
            )
        else:
            st.info("Upload a chest X-ray image to get started.")

    with tab_dashboard:
        st.subheader("Session Statistics")
        history = st.session_state.history

        if not history:
            st.write("No scans processed yet this session. Upload an X-ray in the Predict tab.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Scans processed", len(history))
            col2.metric("Avg. confidence", f"{sum(h['confidence'] for h in history) / len(history) * 100:.1f}%")
            n_pneumonia = sum(1 for h in history if h["label"] != "Normal")
            col3.metric("Flagged pneumonia", n_pneumonia)

            pred_counts = {}
            for h in history:
                pred_counts[h["label"]] = pred_counts.get(h["label"], 0) + 1

            c1, c2 = st.columns(2)
            with c1:
                fig1 = plot_prediction_distribution(pred_counts)
                st.pyplot(fig1)
            with c2:
                fig2 = plot_confidence_histogram([h["confidence"] for h in history])
                st.pyplot(fig2)

        st.divider()
        st.subheader("Model Evaluation (from training run)")
        st.write(
            "Confusion matrix, ROC curves, and Cohen's Kappa from the held-out test set "
            "are generated in `notebooks/Pneumonia_Detection.ipynb`. Export those figures "
            "to `assets/` (e.g. `assets/confusion_matrix.png`) to have them appear here "
            "automatically."
        )
        for fname, caption in [
            ("assets/confusion_matrix.png", "Confusion Matrix"),
            ("assets/roc_curve.png", "ROC Curves"),
        ]:
            if os.path.exists(fname):
                st.image(fname, caption=caption, use_container_width=True)


if __name__ == "__main__":
    main()
