import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Load pre-trained ResNet50
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last two blocks
for name, child in list(model.named_children())[-3:]:  
    for param in child.parameters():
        param.requires_grad = True

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Use different learning rates
optimizer = optim.Adam([
    {'params': model.layer3.parameters(), 'lr': wandb.config.lr_base},
    {'params': model.layer4.parameters(), 'lr': wandb.config.lr_base},
    {'params': model.fc.parameters(), 'lr': wandb.config.lr_fc},
])

criterion = nn.CrossEntropyLoss()

