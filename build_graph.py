import torch
from sklearn.neighbors import NearestNeighbors


# Load extracted CNN features
data = torch.load(
    "gtsrb_features.pt",
    map_location="cpu",
    weights_only=False
)

features = data["features"].float()
labels = data["labels"].long()

print("Features:", features.shape)
print("Labels:", labels.shape)

# Convert features to NumPy
features_numpy = features.numpy()

# Find 5 nearest neighbors for each image
k = 5

knn = NearestNeighbors(
    n_neighbors=k + 1,
    metric="euclidean",
    algorithm="auto",
    n_jobs=-1
)

print("\nFinding nearest neighbors...")
knn.fit(features_numpy)

distances, neighbors = knn.kneighbors(features_numpy)

# Remove each image from its own neighbor list
neighbors = neighbors[:, 1:]

# Create graph edges
source_nodes = torch.arange(
    features.shape[0]
).repeat_interleave(k)

target_nodes = torch.tensor(
    neighbors,
    dtype=torch.long
).reshape(-1)

edge_index = torch.stack(
    [source_nodes, target_nodes],
    dim=0
)

# Save graph
torch.save(
    {
        "edge_index": edge_index,
        "features": features,
        "labels": labels
    },
    "gtsrb_graph.pt"
)

print("\nGraph construction complete.")
print("Number of nodes:", features.shape[0])
print("Features per node:", features.shape[1])
print("Number of edges:", edge_index.shape[1])
print("Edge shape:", edge_index.shape)
print("Saved as: gtsrb_graph.pt")
