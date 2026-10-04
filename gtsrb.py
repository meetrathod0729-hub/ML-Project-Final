import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score

from gtsrb_dataset import train_loader, val_loader, test_loader


# -------------------------
# CNN model
# -------------------------

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


# -------------------------
# Training setup
# -------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -------------------------
# Training
# -------------------------

for epoch in range(10):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(images)

        loss = loss_function(output, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        "Epoch:",
        epoch + 1,
        "Loss:",
        round(total_loss / len(train_loader), 4)
    )


# -------------------------
# Testing
# -------------------------

model.eval()

predictions = []
actual = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        output = model(images)

        predicted = output.argmax(1).cpu()

        predictions.extend(predicted.numpy())
        actual.extend(labels.numpy())


# -------------------------
# Results
# -------------------------

accuracy = accuracy_score(actual, predictions)

f1 = f1_score(
    actual,
    predictions,
    average="macro"
)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")
print("Macro F1:", round(f1, 4))


# -------------------------
# Save model
# -------------------------

torch.save(
    model.state_dict(),
    "gtsrb_cnn.pth"
)

print("Model saved as gtsrb_cnn.pth")