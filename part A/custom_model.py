# Importing libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import 
import torchvision.transforms as T
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


# Building a class for CNN model
class CNNLightningModel(pl.LightningModule):
    def __init__(
        self,
        input_size=(600, 800, 3),
        num_classes=10,
        conv_channels=[16, 32, 64, 64, 64],
        kernel_sizes=[3, 3, 3, 3, 3],
        activation_fn="ReLU",
        dense_neurons=512,
        max_pool_size=2,
        learning_rate=1e-3,
        batch_norm=False,
        dropout=0.2,
    ):
        super(CNNLightningModel, self).__init__()
        self.save_hyperparameters()
        # Map activation function
        activation_map = {
            "ReLU": nn.ReLU,
            "GELU": nn.GELU,
            "SiLU": nn.SiLU,
            "Mish": nn.Mish,
        }
        act_fn = activation_map[activation_fn]
        self.act_fn = act_fn()
        
        self.conv_layers = nn.ModuleList()
        in_channels = input_size[2]
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            layers = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            ]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(act_fn())
            layers.append(nn.MaxPool2d(kernel_size=max_pool_size))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout/2))  # Lower dropout for conv layers
            self.conv_layers.append(nn.Sequential(*layers))
            in_channels = out_channels
            
        # Calculate flattened size
        height, width = input_size[0], input_size[1]
        for _ in range(len(conv_channels)):
            height //= max_pool_size
            width //= max_pool_size
        flatten_size = conv_channels[-1] * height * width
        
        self.fc1 = nn.Linear(flatten_size, dense_neurons)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(dense_neurons, num_classes)
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.act_fn(self.fc1(x))
        x = self.dropout(x)
        return self.fc_out(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_acc": acc}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }