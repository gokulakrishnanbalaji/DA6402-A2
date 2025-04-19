# iNaturalist-12K Image Classification Project

## Name: Gokulakrishnan Balaji
## Roll No: DA24M007

## Overview

This repository contains two Jupyter notebooks (`Part A.ipynb` and `Part B.ipynb`) that implement image classification on the iNaturalist-12K dataset using deep learning techniques. The project explores two approaches: training a custom Convolutional Neural Network (CNN) from scratch and fine-tuning a pre-trained ResNet50 model.

## Dataset

The iNaturalist-12K dataset is used for this project, consisting of images from 10 classes of natural species. The dataset can be downloaded from:\
https://storage.googleapis.com/wandb_datasets/nature_12K.zip

The dataset is split into:

- **Training Set**: 80% of the data (further split into training and validation sets).
- **Validation Set**: 20% of the training data (used for hyperparameter tuning).
- **Test Set**: Provided separately in the dataset for final evaluation.

## Repository Structure

- **Part A.ipynb**: Implements a 5-layer CNN trained from scratch with hyperparameter tuning using Weights & Biases (W&B) sweeps.
- **Part B.ipynb**: Fine-tunes a pre-trained ResNet50 model by unfreezing the last two blocks and the fully connected layer.
- I didn't use any **.py** files as I did my entire work in kaggle notebooks.

## Part A: Training a CNN from Scratch

### Objective

Build and train a 5-layer CNN to classify images from the iNaturalist-12K dataset, optimizing performance through hyperparameter tuning.

### Key Steps

1. **Data Preparation**:

   - Compute input image shape (600x800x3) to design the CNN.
   - Split the training data into 80% training and 20% validation sets using `train_test_split`.
   - Apply data augmentation (random flips, rotations, color jitter) for training.

2. **Model Architecture**:

   - A custom `CNNLightningModel` class is implemented using PyTorch Lightning.
   - The model consists of 5 convolutional layers with configurable channels, kernel sizes, activation functions (ReLU, GELU, SiLU, Mish), batch normalization, dropout, and max pooling.
   - Fully connected layers map the flattened features to 10 output classes.

3. **Hyperparameter Tuning**:

   - W&B sweeps are used to tune parameters like convolutional channels, activation functions, batch normalization, dropout, learning rate, and batch size.
   - The best model is selected based on validation accuracy and saved as a W&B artifact.

4. **Evaluation**:

   - The best model achieves a test accuracy of **41.85%**.
   - Visualizations of correct and incorrect predictions are generated using Matplotlib.

### Dependencies

- PyTorch
- PyTorch Lightning
- torchvision
- Weights & Biases (wandb)
- scikit-learn
- matplotlib
- numpy
- PIL

## Part B: Fine-Tuning ResNet50

### Objective

Fine-tune a pre-trained ResNet50 model on the iNaturalist-12K dataset to achieve higher accuracy compared to the custom CNN.

### Key Steps

1. **Data Preparation**:

   - Similar to Part A, the training data is split into 80% training and 20% validation sets.
   - Images are resized to 224x224 and normalized for compatibility with ResNet50.

2. **Model Setup**:

   - Load a pre-trained ResNet50 model from torchvision.
   - Replace the fully connected layer to output 10 classes.
   - Freeze all layers except the last two convolutional blocks (layer3 and layer4) and the fully connected layer.
   - Use different learning rates: 1e-4 for convolutional layers and 1e-3 for the fully connected layer.

3. **Training**:

   - Train for 15 epochs using the Adam optimizer and CrossEntropyLoss.
   - Log training and validation metrics (loss, accuracy) to W&B.

4. **Evaluation**:

   - The fine-tuned model achieves a test accuracy of **77.3%**, significantly outperforming the custom CNN.
   - Test loss is reported as 1.279.

### Dependencies

- PyTorch
- torchvision
- Weights & Biases (wandb)
- scikit-learn


## Results

- **Part A (Custom CNN)**:

  - Test Accuracy: **41.85%**
  - Best model configuration: Convolutional channels \[16, 64, 128, 256, 256\], Mish activation, batch normalization enabled, no dropout.

- **Part B (ResNet50)**:

  - Test Accuracy: **77.3%**
  - Test Loss: **1.279**
  - Strategy: Unfreeze last two convolutional blocks and fully connected layer.

## Visualizations

- **Part A**: A 10x3 grid of test set predictions (correct and incorrect) is plotted to analyze model performance per class.
- **Part B**: No visualizations are included, but test metrics are logged.
