"""
Standalone training script, extracted and cleaned up from
notebooks/Pneumonia_Detection.ipynb.

Expects the Kaggle "Chest X-Ray Images (Pneumonia)" dataset extracted at
dataset/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg
(PNEUMONIA filenames containing "bacteria" or "virus" are split into the
Bacterial / Viral classes, matching the notebook's labeling logic.)

Usage:
    python train.py --backbone resnet50 --epochs 7 --data_dir dataset/chest_xray
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

from utils.model_def import CNNClassifier, CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, get_device

LABEL_MAP = {"NORMAL": 0, "BACTERIAL": 1, "VIRAL": 2}


def build_dataframe(data_dir: str) -> pd.DataFrame:
    paths, labels = [], []
    for phase in ["train", "val", "test"]:
        for cat in ["NORMAL", "PNEUMONIA"]:
            folder = os.path.join(data_dir, phase, cat)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if not fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    continue
                if cat == "NORMAL":
                    label = LABEL_MAP["NORMAL"]
                elif "bacteria" in fname.lower():
                    label = LABEL_MAP["BACTERIAL"]
                elif "virus" in fname.lower():
                    label = LABEL_MAP["VIRAL"]
                else:
                    label = LABEL_MAP["BACTERIAL"]  # default untagged pneumonia to bacterial
                paths.append(os.path.join(folder, fname))
                labels.append(label)
    return pd.DataFrame({"path": paths, "label": labels})


class PneumoniaDS(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.tfms = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row.path).convert("RGB"))
        img = self.tfms(image=img)["image"]
        return img, row.label


def get_transforms():
    train_tfms = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    val_tfms = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return train_tfms, val_tfms


def train(backbone_name: str, epochs: int, data_dir: str, batch_size: int, lr: float, out_path: str):
    device = get_device()
    print(f"Using device: {device}")

    df = build_dataframe(data_dir)
    print(df["label"].value_counts().rename(index={0: "Normal", 1: "Bacterial", 2: "Viral"}))

    df_train, df_temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=1 / 3, stratify=df_temp["label"], random_state=42)
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    train_tfms, val_tfms = get_transforms()
    train_loader = DataLoader(PneumoniaDS(df_train, train_tfms), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(PneumoniaDS(df_val, val_tfms), batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(PneumoniaDS(df_test, val_tfms), batch_size=batch_size, shuffle=False, num_workers=2)

    model = CNNClassifier(backbone_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_acc = 0.0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item()
                _, pred = torch.max(out, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_path)
            print(f"  -> New best model saved to {out_path} (val_acc={acc:.4f})")

    # Final test-set evaluation with the best checkpoint
    model.load_state_dict(torch.load(out_path, map_location=device))
    model.eval()
    preds, lbls = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            _, pred = torch.max(out, 1)
            preds.extend(pred.cpu().numpy())
            lbls.extend(labels.numpy())

    print("\nTest set performance:")
    print(classification_report(lbls, preds, target_names=CLASS_NAMES))
    print(f"Test accuracy: {accuracy_score(lbls, preds):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the pneumonia detection model.")
    parser.add_argument("--backbone", default="resnet50", help="timm backbone (e.g. resnet50, efficientnet_b0)")
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--data_dir", default="dataset/chest_xray")
    parser.add_argument("--out_path", default="model/pneumonia_model.pt")
    args = parser.parse_args()

    train(args.backbone, args.epochs, args.data_dir, args.batch_size, args.lr, args.out_path)
