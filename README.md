# AI-Assisted Pneumonia Detection from Chest X-Rays

A deep learning system that classifies chest X-rays as **Normal**, **Bacterial Pneumonia**,
or **Viral Pneumonia**, with Grad-CAM explainability, a Streamlit diagnostic-support
interface, and downloadable PDF reports.

> **Note on scope:** this is a 3-class problem (Normal / Bacterial / Viral), built with
> **PyTorch + timm** transfer learning (ResNet50 and EfficientNet-B0 backbones), not a
> binary Normal-vs-Pneumonia TensorFlow model. The classifier was tuned with a
> **SAM (Sharpness-Aware Minimization) optimizer** and **OneCycleLR** scheduling, and
> evaluated with classification report, confusion matrix, ROC/AUC, and Cohen's Kappa.

## Features

- **Transfer learning** on ResNet50 / EfficientNet-B0 (timm), fine-tuned end-to-end
- **Hyperparameter tuning**: SAM optimizer vs. OneCycleLR, compared head-to-head
- **Explainable AI**: Grad-CAM heatmaps showing which lung regions drove each prediction
- **Comprehensive evaluation**: accuracy, classification report, confusion matrix, ROC/AUC, Cohen's Kappa
- **Interactive web app** (Streamlit): upload an X-ray, see the prediction, confidence, and Grad-CAM overlay
- **PDF report generation**: original image, Grad-CAM overlay, prediction, confidence, timestamp, medical disclaimer
- **Session dashboard**: scans processed, prediction distribution, confidence histogram

## Project Structure

```
AI-Pneumonia-Detection/
├── app.py                  # Streamlit web app
├── train.py                # Training script (standalone, cleaned from notebook)
├── predict.py               # CLI inference on a single image
├── requirements.txt
├── README.md
├── LICENSE
│
├── model/
│   └── pneumonia_model.pt   # Trained weights (generate via train.py)
│
├── utils/
│   ├── model_def.py          # Shared CNNClassifier architecture
│   ├── preprocess.py         # Inference-time image preprocessing
│   ├── gradcam.py             # Grad-CAM implementation
│   ├── report.py              # PDF report generation
│   └── visualization.py       # Plots: triptych, confusion matrix, ROC, histograms
│
├── dataset/                  # Kaggle chest_xray dataset (gitignored)
├── assets/                   # Exported evaluation figures for the dashboard
├── screenshots/               # App screenshots for this README
└── notebooks/
    └── Pneumonia_Detection.ipynb   # Full training/evaluation notebook
```

## Setup

```bash
git clone <your-repo-url>
cd AI-Pneumonia-Detection
pip install -r requirements.txt
```

### 1. Get the dataset

Uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
Kaggle dataset. Download and extract it to `dataset/chest_xray/` so you have:

```
dataset/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg
```

### 2. Train

```bash
python train.py --backbone resnet50 --epochs 7 --data_dir dataset/chest_xray --out_path model/pneumonia_model.pt
```

### 3. Run a single prediction from the command line

```bash
python predict.py --image path/to/xray.jpg --weights model/pneumonia_model.pt
```

### 4. Launch the web app

```bash
streamlit run app.py
```

## Model Performance

_Fill in after running `train.py` / the notebook's evaluation cells:_

| Model | Optimizer | Test Accuracy | Cohen's Kappa |
|---|---|---|---|
| ResNet50 | AdamW | — | — |
| ResNet50 | SAM | — | — |
| ResNet50 | OneCycleLR | — | — |
| EfficientNet-B0 | AdamW | — | — |

Export your confusion matrix and ROC plots from the notebook to `assets/confusion_matrix.png`
and `assets/roc_curve.png` — the Streamlit dashboard tab will pick them up automatically.

## Tech Stack

Python · PyTorch · timm · Albumentations · OpenCV · scikit-learn · Streamlit · Grad-CAM · ReportLab

## Medical Disclaimer

This project is an educational/portfolio AI-assisted decision-support demo. It is **not**
a certified medical device, has **not** been clinically validated, and must never be used
for real diagnosis or treatment decisions. Always consult a licensed radiologist or physician.

## Resume Bullet

> Developed an AI-assisted pneumonia detection system using transfer learning (ResNet50/EfficientNet-B0)
> on chest X-ray images, integrating explainable AI (Grad-CAM), a SAM/OneCycleLR-tuned training pipeline,
> and a Streamlit diagnostic interface with automated PDF report generation.
