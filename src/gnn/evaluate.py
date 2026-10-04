import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from src.gnn.model import GCN


GRAPH_PATH = "data/processed/graph.pt"
MODEL_PATH = "experiments/checkpoints/gcn.pth"


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    data = torch.load(
        GRAPH_PATH,
        weights_only=False
    )

    data = data.to(device)

    model = GCN(
        input_features=data.x.shape[1],
        hidden_features=64,
        num_classes=58
    ).to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    with torch.no_grad():

        output = model(
            data.x,
            data.edge_index
        )

        predictions = output.argmax(
            dim=1
        )

    y_true = data.y[
        data.test_mask
    ].cpu().numpy()

    y_pred = predictions[
        data.test_mask
    ].cpu().numpy()

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(
        figsize=(10, 10)
    )

    plt.imshow(matrix)

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.title(
        "GNN Confusion Matrix"
    )

    plt.colorbar()

    plt.savefig(
        "experiments/results/gcn_confusion_matrix.png"
    )

    print(
        "Confusion matrix saved."
    )


if __name__ == "__main__":
    main()