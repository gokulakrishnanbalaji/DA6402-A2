import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
import torchvision.transforms as T
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloader import *


# Initialize W&B API
api = wandb.Api()

# Replace with your actual values
entity = "da24m007-iit-madras"
project = "DL-A2-V2"
artifact_name = "best-cnn-model"

# Get all versions of the model artifact
artifact_versions = api.artifact_versions(
    type_name="model",
    name=f"{entity}/{project}/{artifact_name}"
)

# Store artifact and val_acc
artifact_scores = []

for artifact in artifact_versions:
    val_acc = artifact.metadata.get("val_acc", None)
    if val_acc is not None:
        artifact_scores.append((val_acc, artifact))

# Sort artifacts by val_acc
artifact_scores.sort(key=lambda x: x[0], reverse=True)

if len(artifact_scores) == 0:
    print("❌ No artifacts found with val_acc in metadata.")
else:
    # Best one
    best_val_acc, best_artifact = artifact_scores[0]
    print(f"✅ Best model: {best_artifact.name} with val_acc = {best_val_acc}")

    # Download it
    best_model_dir = best_artifact.download()
    ckpt_path = os.path.join(best_model_dir, "best-model.ckpt")

    # Load it
    model = CNNLightningModel.load_from_checkpoint(ckpt_path)
   

    print("🎯 Model loaded and ready for prediction!")

import torch
from sklearn.metrics import accuracy_score

all_preds = []
all_labels = []

model.eval()
device = next(model.parameters()).device  # Get model's device

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
print(f"✅ Test Accuracy: {test_acc:.4f}")


import matplotlib.pyplot as plt
import numpy as np
import torch

# Move model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prepare a structure to store predictions
class_predictions = {i: {'correct': [], 'incorrect': []} for i in range(10)}  # adjust if more classes

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        for img, pred, label in zip(images, preds, labels):
            img_cpu = img.cpu()
            if pred == label:
                class_predictions[label.item()]['correct'].append((img_cpu, pred.item()))
            else:
                class_predictions[label.item()]['incorrect'].append((img_cpu, pred.item()))

# Plot 10x3 grid
fig, axes = plt.subplots(10, 3, figsize=(12, 25))
fig.suptitle(" Test Set Predictions by Best Model", fontsize=20, weight='bold')

for i in range(10):
    corrects = class_predictions[i]['correct']
    incorrects = class_predictions[i]['incorrect']
    plotted = 0

    # First column: correct if exists, else fallback to incorrect
    if corrects:
        img, pred = corrects[0]
        title = f"Pred: {class_names[pred]}\n True: {class_names[i]}"
    elif incorrects:
        img, pred = incorrects[0]
        title = f" Pred: {class_names[pred]}\n True: {class_names[i]}"
        incorrects = incorrects[1:]  # remove the one used
    else:
        img, pred = None, None

    if img is not None:
        axes[i, 0].imshow(np.clip(img.permute(1, 2, 0).numpy(), 0, 1))
        axes[i, 0].set_title(title, fontsize=8)
        axes[i, 0].axis("off")
        plotted += 1
    else:
        axes[i, 0].axis("off")

    # Next two columns: other incorrect predictions (if available)
    for j in range(2):
        if j < len(incorrects):
            img, pred = incorrects[j]
            axes[i, j + 1].imshow(np.clip(img.permute(1, 2, 0).numpy(), 0, 1))
            axes[i, j + 1].set_title(f" Pred: {class_names[pred]}\n True: {class_names[i]}", fontsize=8)
            axes[i, j + 1].axis("off")
        else:
            axes[i, j + 1].axis("off")

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()