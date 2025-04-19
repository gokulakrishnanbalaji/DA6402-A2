# importing libraries

import os
import shutil
from sklearn.model_selection import train_test_split


for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir,class_name)

    # error handling in case the class_path does not exist
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    train_imgs, val_imgs = train_test_split(images,test_size=0.2, random_state=42)

    # create a new directory for storing val_imgs
    os.makedirs(os.path.join(new_val_dir, class_name), exist_ok = True)
    os.makedirs(os.path.join(new_train_dir, class_name), exist_ok = True)

    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(new_val_dir, class_name, img))

    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(new_train_dir, class_name, img))


from torch.utils.data import DataLoader
from torchvision import datasets

# Datasets
train_dataset = datasets.ImageFolder(root=new_train_dir, transform=get_transforms())
val_dataset = datasets.ImageFolder(root=new_val_dir, transform=get_transforms())
test_dataset = datasets.ImageFolder(root='/kaggle/input/nature-12k-new/inaturalist_12K/val', transform=get_transforms())

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)