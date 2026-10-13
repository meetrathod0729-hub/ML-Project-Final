
import torch

data = torch.load(
    "gtsrb_graph.pt",
    map_location="cpu",
    weights_only=False
)

labels = data["labels"]
edge_index = data["edge_index"]

source = edge_index[0]
target = edge_index[1]

same_class = labels[source] == labels[target]
same_class_percentage = same_class.float().mean().item() * 100

print("Graph quality check")
print("-------------------")
print("Nodes:", len(labels))
print("Edges:", edge_index.shape[1])
print(
    "Neighbors with the same class:",
    f"{same_class_percentage:.2f}%"
)

# Check how many nodes have at least 4 of 5
# neighbors belonging to the same class.
same_class_per_node = torch.zeros(
    len(labels), dtype=torch.long
)
same_class_per_node.scatter_add_(
    0, source, same_class.long()
)

mostly_same = (same_class_per_node >= 4).float().mean().item() * 100

print(
    "Nodes with at least 4 same-class neighbors:",
    f"{mostly_same:.2f}%"
)
