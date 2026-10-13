import os
import torch
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.gnn.model import GCN


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load graph and split
    graph = torch.load(
        "gtsrb_graph.pt",
        map_location="cpu",
        weights_only=False
    )

    split = torch.load(
        "gtsrb_gcn_split.pt",
        map_location="cpu",
        weights_only=False
    )

    x = graph["features"].float().to(device)
    y = graph["labels"].long().to(device)
    edges = graph["edge_index"].long().to(device)

    test_mask = split["heldout_mask"].to(device)

    # Load trained GCN
    model = GCN(
        input_features=128,
        hidden_features=128,
        num_classes=43
    ).to(device)

    model.load_state_dict(
        torch.load(
            "gtsrb_gcn.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    # Predict classes
    with torch.no_grad():
        output = model(x, edges)
        pred = output.argmax(dim=1)

    actual = y[test_mask].cpu().numpy()
    predicted = pred[test_mask].cpu().numpy()

    # Calculate results
    acc = accuracy_score(actual, predicted)
    f1 = f1_score(
        actual,
        predicted,
        average="macro",
        zero_division=0
    )

    print("\nGCN Results")
    print("Accuracy:", round(acc * 100, 2), "%")
    print("Macro F1:", round(f1, 4))
    print("Images evaluated:", len(actual))

    # Create confusion matrix
    matrix = confusion_matrix(
        actual,
        predicted,
        labels=list(range(43))
    )

    # Save only as CSV
    os.makedirs("experiments/results", exist_ok=True)

    table = pd.DataFrame(
        matrix,
        index=[f"True_{i}" for i in range(43)],
        columns=[f"Pred_{i}" for i in range(43)]
    )

    table.to_csv(
        "experiments/results/gcn_confusion_matrix.csv",
        index_label="Actual_Class"
    )

    print("Confusion matrix saved to experiments/results/gcn_confusion_matrix.csv")


if __name__ == "__main__":
    main()
