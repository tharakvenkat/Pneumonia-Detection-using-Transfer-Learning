# 🫁 Pneumonia Detection using Transfer Learning

A deep learning framework for **multi-class pneumonia classification** from chest X-ray images using **transfer learning** with **ResNet50** and **EfficientNet-B0**. The project explores both **feature extraction** and **end-to-end fine-tuning**, incorporating **Grad-CAM explainability** and comprehensive performance evaluation.

> **Project Scope**
>
> This project performs **3-class classification** of chest X-ray images into:
> - Normal
> - Bacterial Pneumonia
> - Viral Pneumonia
>
> Multiple transfer learning strategies were evaluated using **PyTorch** and the **timm** library, including feature extraction with Logistic Regression and fine-tuned CNNs. Model performance was assessed using **Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix,** and **Cohen's Kappa**.

---

# 🚀 Features

- Multi-class chest X-ray classification (Normal, Bacterial, Viral)
- Transfer learning using **ResNet50** and **EfficientNet-B0**
- Comparison of **feature extraction** and **fine-tuning** approaches
- Hyperparameter optimization using **SAM Optimizer** and **OneCycleLR**
- Explainable AI using **Grad-CAM**
- Comprehensive evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
  - ROC-AUC
  - Cohen's Kappa
- Comparative analysis of multiple transfer learning strategies

---

# 📂 Project Structure

```
Pneumonia-Detection-using-Transfer-Learning/
│
├── README.md
├── LICENSE
├── requirements.txt
├── Pneumonia_Detection.ipynb
│
├── assets/
│   ├── confusion_matrix.png
│   ├── gradcam_example.png
│   ├── metrics.csv
│   ├── model_comparison.png
│   ├── roc_curve.png
│   └── training_curves.png
```

---

# 📊 Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle.

Dataset Classes:

- Normal
- Bacterial Pneumonia
- Viral Pneumonia

Images are divided into:

- Training Set
- Validation Set
- Test Set

---

# 🏗️ Models Evaluated

The following transfer learning approaches were implemented and compared:

| Model | Strategy |
|--------|----------|
| ResNet50 + Logistic Regression | Feature Extraction |
| EfficientNet-B0 + Logistic Regression | Feature Extraction |
| ResNet50 | End-to-End Fine-Tuning |
| EfficientNet-B0 | End-to-End Fine-Tuning |

---

# 📈 Model Performance

| Model | Training Strategy | Test Accuracy |
|--------|-------------------|--------------:|
| ResNet50 + Logistic Regression | Feature Extraction | **75.1%** |
| EfficientNet-B0 + Logistic Regression | Feature Extraction | **75.4%** |
| EfficientNet-B0 | Fine-Tuning | **81.4%** |
| **ResNet50** | **Fine-Tuning** | **85.3%** |

The fine-tuned **ResNet50** achieved the highest overall performance among all evaluated models.

---

# 📷 Results

## Model Comparison

![Model Comparison](assets/model_comparison.png)

---

## Training Curves

![Training Curves](assets/training_curves.png)

---

## Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

---

## ROC Curve

![ROC Curve](assets/roc_curve.png)

---

## Grad-CAM Visualization

Grad-CAM was used to visualize the image regions that contributed most to the model's predictions, improving interpretability of the deep learning model.

![Grad-CAM](assets/gradcam_example.png)

---

# 🛠️ Tech Stack

- Python
- PyTorch
- timm
- Albumentations
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn
- TorchMetrics

---

# ▶️ Running the Project

Clone the repository

```bash
git clone https://github.com/<your-username>/Pneumonia-Detection-using-Transfer-Learning.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Open the notebook

```
Pneumonia_Detection.ipynb
```

Run all cells sequentially in Google Colab or Jupyter Notebook.

---

# 📌 Key Highlights

- Implemented and compared **four transfer learning configurations**
- Achieved over **85% test accuracy** using a fine-tuned ResNet50
- Improved model interpretability through **Grad-CAM**
- Performed extensive evaluation beyond simple accuracy using multiple performance metrics
- Compared feature extraction and fine-tuning strategies for medical image classification

---

# ⚠️ Medical Disclaimer

This project is intended for **educational and research purposes only**. It is **not** a certified medical device and should **not** be used for clinical diagnosis or treatment decisions.

---

# 👨‍💻 Author

**Tharak Venkat Imadabattuni**

Bachelor of Technology in Computer Science and Engineering
