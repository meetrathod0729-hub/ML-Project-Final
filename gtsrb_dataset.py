import os
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = r"C:\Users\Meet Rathod\Desktop\ML Lab Project datasets\GTSRB"

class GTSRB(Dataset):

    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = os.path.join(DATA_PATH, row["Path"])
        image = Image.open(image_path).convert("RGB")

        label = int(row["ClassId"])

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

train_data = GTSRB(
    os.path.join(DATA_PATH, "Train.csv"),
    transform
)

test_data = GTSRB(
    os.path.join(DATA_PATH, "Test.csv"),
    transform
)

train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size

train_data, val_data = random_split(
    train_data,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_data, batch_size=64, shuffle=True
)

val_loader = DataLoader(
    val_data, batch_size=64
)

test_loader = DataLoader(
    test_data, batch_size=64
)

print("Training images:", len(train_data))
print("Validation images:", len(val_data))
print("Test images:", len(test_data))