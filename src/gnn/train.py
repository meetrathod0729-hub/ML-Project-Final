import torch

from src.gnn.model import GCN


GRAPH_PATH = "data/processed/graph.pt"
MODEL_PATH = "experiments/checkpoints/gcn.pth"

EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_FEATURES = 64


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # Load graph
    data = torch.load(
        GRAPH_PATH,
        weights_only=False
    )

    data = data.to(device)

    print("Nodes:", data.x.shape[0])
    print("Features:", data.x.shape[1])
    print("Edges:", data.edge_index.shape[1])

    # Create model
    model = GCN(
        input_features=data.x.shape[1],
        hidden_features=HIDDEN_FEATURES,
        num_classes=58
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = torch.nn.CrossEntropyLoss()

    # Training
    for epoch in range(EPOCHS):

        model.train()

        optimizer.zero_grad()

        output = model(
            data.x,
            data.edge_index
        )

        loss = criterion(
            output[data.train_mask],
            data.y[data.train_mask]
        )

        loss.backward()

        optimizer.step()

        # Validation
        model.eval()

        with torch.no_grad():

            predictions = output.argmax(dim=1)

            correct = (
                predictions[data.val_mask]
                == data.y[data.val_mask]
            ).sum()

            total = data.val_mask.sum()

            validation_accuracy = (
                correct.float() / total
            )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss: {loss.item():.4f} | "
            f"Val Accuracy: "
            f"{validation_accuracy.item():.4f}"
        )

    # Save model
    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print("\nModel saved to:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()