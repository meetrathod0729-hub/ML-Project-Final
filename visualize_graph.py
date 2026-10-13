
import torch
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA

data = torch.load(
    "gtsrb_graph.pt",
    map_location="cpu",
    weights_only=False
)

features = data["features"]
labels = data["labels"]
edge_index = data["edge_index"]

# Choose the most common class among the first 300 nodes
sample_size = min(300, len(labels))
counts = torch.bincount(labels[:sample_size], minlength=43)
class_id = int(counts.argmax())

# Select 30 images belonging to that class
indices = torch.where(labels == class_id)[0][:100]
selected = set(indices.tolist())

if len(selected) < 2:
    raise ValueError("Not enough images found for this class.")

# Keep only edges between the selected images
G = nx.DiGraph()
G.add_nodes_from(selected)

for s, t in edge_index.T.tolist():
    if s in selected and t in selected:
        G.add_edge(s, t)

# Place nodes using their CNN features
subset_features = features[indices].numpy()
points = PCA(n_components=2).fit_transform(subset_features)
pos = {int(node): points[i] for i, node in enumerate(indices)}

plt.figure(figsize=(10, 8))

nx.draw_networkx_edges(
    G, pos,
    edge_color="gray",
    alpha=0.7,
    arrows=False,
    width=1.5
)

nx.draw_networkx_nodes(
    G, pos,
    node_color="royalblue",
    node_size=180
)

nx.draw_networkx_labels(
    G,
    pos,
    labels={
        int(node): str(i)
        for i, node in enumerate(indices.tolist())
    },
    font_size=8
)

plt.title(f"GTSRB kNN Graph — Class {class_id} (100 Images)")
plt.axis("off")
plt.tight_layout()
plt.savefig(
    "experiments/results/gtsrb_graph_class.png",
    dpi=300
)
plt.show()

print("Class:", class_id)
print("Nodes displayed:", G.number_of_nodes())
print("Connections displayed:", G.number_of_edges())
print("Saved to experiments/results/gtsrb_graph_class.png")
