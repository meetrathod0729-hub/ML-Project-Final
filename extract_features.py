import torch
from torch import nn

from gtsrb_dataset import train_loader


# Same CNN architecture as gtsrb.py
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


# Load trained CNN
model.load_state_dict(torch.load("gtsrb_cnn.pth", map_location=device))

model.eval()


# Remove the final 43-class layer
feature_extractor = nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor = feature_extractor.to(device)


# Store features and labels
all_features = []
all_labels = []

print("Extracting features...")


with torch.no_grad():

    for images, labels in train_loader:

        images = images.to(device)

        features = feature_extractor(images)

        all_features.append(features.cpu())
        all_labels.append(labels)

        
        # Start with 2,000 images
        if sum(x.size(0) for x in all_features) >= 2000:
            break


# Combine batches
features = torch.cat(all_features, dim=0)
labels = torch.cat(all_labels, dim=0)


# Keep exactly 2,000 samples
features = features[:2000]
labels = labels[:2000]


# Save features
torch.save(
    {
        "features": features,
        "labels": labels
    },
    "gtsrb_features.pt"
)


print("Feature extraction complete.")
print("Feature shape:", features.shape)
print("Label shape:", labels.shape)
print("Saved as: gtsrb_features.pt")