import torch
from sklearn.neighbors import NearestNeighbors


# Load extracted CNN features
data = torch.load("gtsrb_features.pt")

features = data["features"]
labels = data["labels"]


print("Features:", features.shape)
print("Labels:", labels.shape)


# Convert features to NumPy
features_numpy = features.numpy()


# Find 5 nearest neighbors for each image
k = 5

knn = NearestNeighbors(
    n_neighbors=k + 1,
    metric="euclidean"
)

knn.fit(features_numpy)

distances, neighbors = knn.kneighbors(features_numpy)


# Remove the image itself from its neighbors
neighbors = neighbors[:, 1:]
distances = distances[:, 1:]


# Create graph edges
source_nodes = []
target_nodes = []

for i in range(len(neighbors)):

    for j in range(k):

        source_nodes.append(i)
        target_nodes.append(neighbors[i][j])


# Convert edges to tensors
edge_index = torch.tensor(
    [source_nodes, target_nodes],
    dtype=torch.long
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


print("Graph construction complete.")
print("Number of nodes:", features.shape[0])
print("Number of edges:", edge_index.shape[1])
print("Edge shape:", edge_index.shape)
print("Saved as: gtsrb_graph.pt")