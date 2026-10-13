import os
import torch
from sklearn.model_selection import train_test_split

from src.gnn.model import GCN

GRAPH_PATH = "gtsrb_graph.pt"
MODEL_PATH = "gtsrb_gcn.pth"
SPLIT_PATH = "gtsrb_gcn_split.pt"

EPOCHS = 200
LEARNING_RATE = 0.001
HIDDEN_FEATURES = 128
NUM_CLASSES = 43
SEED = 42


def main():
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False
    )

    features = graph["features"].float()
    labels = graph["labels"].long()
    edge_index = graph["edge_index"].long()

    print("Nodes:", features.shape[0])
    print("Features per node:", features.shape[1])
    print("Edges:", edge_index.shape[1])
    print("Classes found:", torch.unique(labels).numel())

    if labels.min().item() < 0 or labels.max().item() >= NUM_CLASSES:
        raise ValueError("Labels must be in the range 0 to 42.")

    indices = list(range(len(labels)))

    train_indices, remaining = train_test_split(
        indices,
        test_size=0.30,
        random_state=SEED,
        stratify=labels.numpy()
    )

    val_indices, heldout_indices = train_test_split(
        remaining,
        test_size=0.50,
        random_state=SEED,
        stratify=labels[remaining].numpy()
    )

    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    val_mask = torch.zeros(len(labels), dtype=torch.bool)
    heldout_mask = torch.zeros(len(labels), dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    heldout_mask[heldout_indices] = True

    torch.save(
        {
            "train_mask": train_mask,
            "val_mask": val_mask,
            "heldout_mask": heldout_mask
        },
        SPLIT_PATH
    )

    features = features.to(device)
    labels = labels.to(device)
    edge_index = edge_index.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    model = GCN(
        input_features=features.shape[1],
        hidden_features=HIDDEN_FEATURES,
        num_classes=NUM_CLASSES
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = torch.nn.CrossEntropyLoss()
    best_val_accuracy = -1.0

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        output = model(features, edge_index)

        loss = criterion(
            output[train_mask],
            labels[train_mask]
        )

        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            output = model(features, edge_index)
            predictions = output.argmax(dim=1)

            val_accuracy = (
                predictions[val_mask] == labels[val_mask]
            ).float().mean().item()

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss: {loss.item():.4f} | "
            f"Validation Accuracy: {val_accuracy * 100:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_PATH)

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
    print("Model saved as:", MODEL_PATH)
    print("Split saved as:", SPLIT_PATH)


if __name__ == "__main__":
    main()
