import torch
from torch import nn

from gtsrb_dataset import train_loader


# Same CNN used in gtsrb.py
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),

    nn.Linear(128 * 4 * 4, 128),
    nn.ReLU(),

    nn.Linear(128, 43)
)


# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Load the trained CNN
model.load_state_dict(
    torch.load(
        "gtsrb_cnn.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

# Remove the final classification layer
feature_model = nn.Sequential(
    *list(model.children())[:-1]
).to(device)

all_features = []
all_labels = []

print("Extracting features from the full training split...")
print("Device:", device)

with torch.no_grad():
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)

        features = feature_model(images)

        all_features.append(features.cpu())
        all_labels.append(labels)

        if (i + 1) % 100 == 0:
            print("Batches processed:", i + 1)

# Combine all batches
features = torch.cat(all_features, dim=0)
labels = torch.cat(all_labels, dim=0)

# Save all features and labels
torch.save(
    {
        "features": features,
        "labels": labels
    },
    "gtsrb_features.pt"
)

print("\nFeature extraction complete.")
print("Number of samples:", features.shape[0])
print("Features per sample:", features.shape[1])
print("Labels:", labels.shape[0])
print("Saved as: gtsrb_features.pt")
