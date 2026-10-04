import torch


# Load graph
data = torch.load("gtsrb_graph.pt")

features = data["features"]
labels = data["labels"]
edge_index = data["edge_index"]


# Basic information
print("Graph validation")
print("----------------")

print("Number of nodes:", features.shape[0])
print("Features per node:", features.shape[1])
print("Number of labels:", labels.shape[0])
print("Number of edges:", edge_index.shape[1])


# Check feature and label count
print("\nFeature-label check:")

if features.shape[0] == labels.shape[0]:
    print("PASS: Every node has one label.")
else:
    print("FAIL: Feature and label counts do not match.")


# Check edge format
print("\nEdge check:")

if edge_index.shape[0] == 2:
    print("PASS: Edge index has correct shape.")
else:
    print("FAIL: Edge index shape is incorrect.")


# Check that edge node IDs are valid
if edge_index.min() >= 0 and edge_index.max() < features.shape[0]:
    print("PASS: All edge node IDs are valid.")
else:
    print("FAIL: Invalid node ID found in edges.")


# Check number of neighbors
expected_edges = features.shape[0] * 5

print("\nNeighbor check:")

if edge_index.shape[1] == expected_edges:
    print("PASS: Every node has 5 neighbors.")
else:
    print("WARNING: Edge count is different from expected.")


print("\nGraph validation complete.")