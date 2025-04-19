from custom_model import *
from dataloader import *

# Set up wandb logger with python lightning

from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project='DL-A2-V2')

# Data augmentation transforms
def get_transforms(data_augmentation=False):
    IMAGE_SIZE = (224, 224) 
    if data_augmentation:
        return T.Compose([
            T.Resize(IMAGE_SIZE),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)  # normalize to [-1, 1]
        ])
    return T.Compose([
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)  # normalize to [-1, 1]
    ])

# Training function for sweep

def train_sweep():
    run = wandb.init()
    config = wandb.config

    print(config)

    # Datasets
    train_dataset = datasets.ImageFolder(root=new_train_dir, transform=get_transforms(bool(config.data_augmentation)))
    val_dataset = datasets.ImageFolder(root=new_val_dir, transform=get_transforms(bool(config.data_augmentation)))

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Initialize model
    model = CNNLightningModel(
        input_size=(224, 224, 3),
        conv_channels=config.conv_channels,
        kernel_sizes=[3] * len(config.conv_channels),
        activation_fn=config.activation_fn,
        dense_neurons=512,
        max_pool_size=2,
        learning_rate=config.learning_rate,
        batch_norm=config.batch_norm,
        dropout=config.dropout,
    )

    # W&B logger
    wandb_logger = WandbLogger(log_model=False)

    # Callback to save best model based on val_acc
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",               # Must match your `log()` key in validation step
        filename="best-model",          # Saves as "best-model.ckpt"
        save_top_k=1,
        mode="max",
        save_weights_only=False
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    # Train
    trainer.fit(model, train_dataloader, val_dataloader)

    # Log the best model checkpoint as W&B artifact
    best_model_path = checkpoint_callback.best_model_path

    if best_model_path:
        artifact = wandb.Artifact("best-cnn-model", type="model")
        artifact.add_file(best_model_path)

        # Optionally include val_acc in metadata
        val_acc = wandb_logger.experiment.summary.get("val_acc")
        if val_acc is not None:
            artifact.metadata = {"val_acc": val_acc}

        run.log_artifact(artifact)

    run.finish()

# Sweep configuration
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "conv_channels": {
            "values": [
                [32, 32, 32,32,32],
                [64, 64, 64,64,64],
                [16, 32, 64,128,256],
                [32, 64, 128, 64, 32],
                [256, 128, 64, 32, 16],  # Halving channels
                [32, 64, 64, 128, 128], 
                [16, 64, 128, 256, 256],  # Doubling channels
                [256, 128, 64, 32, 32]   # Decreasing channels
            ]
        },
        "activation_fn": {"values": ["ReLU", "GELU", "SiLU", "Mish"]},
        "data_augmentation": {"values": [True, False]},
        "batch_norm": {"values": [True, False]},
        "dropout": {"values": [0.0, 0.2, 0.3, 0.4]},
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        "batch_size": {"values": [16, 32, 64]},
    },
}

# Initialize and run sweep
sweep_id = wandb.sweep(sweep_config, project = 'DL-A2-V2')
wandb.agent(sweep_id, function=train_sweep, count=20) 