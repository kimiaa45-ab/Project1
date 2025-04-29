import yaml
import json
import os
import torch
import copy
from setting import initialize_settings
from gnn_model import gnn_model
from AR_model import AR_model
import numpy as np  # Added for debugging connectivity shapes

# Load configuration from YAML file
config_path = "configs/config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Extract configuration parameters
num_epochs = config["model"]["num_epochs"]
num_samples = config["model"]["num_samples"]
num_layers = config["model"]["num_layers"]
hidden_dim = config["model"]["hidden_dim"]
cpu_hidden_dim = config["model"]["cpu_hidden_dim"]
device = config["model"]["device"]

# Determine device and set dim accordingly
if device == "auto":
    if torch.cuda.is_available():
        device = "cuda"
        dim = hidden_dim
    else:
        device = "cpu"
        dim = cpu_hidden_dim
else:
    device = device.lower()
    dim = hidden_dim if device == "cuda" else cpu_hidden_dim

# Initialize or load parameters from settings.json
parameters = initialize_settings(
    config_path="configs/config.yaml", settings_path="configs/setting.json"
)

# Initialize lists to store data
microservices = []
microservices_edges = []
computingnodes = []
computingnodes_edges = []


# Modify the componentConnections matrix
def modify_connections(matrix):
    if not matrix or not all(isinstance(row, list) for row in matrix):
        raise ValueError("Input must be a non-empty list of lists")
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("Input must be a square matrix")
    if not all(all(val in [0, 1] for val in row) for row in matrix):
        raise ValueError("Matrix must contain only 0 or 1")

    modified_matrix = copy.deepcopy(matrix)
    for i in range(n):
        for j in range(n):
            if i == j:
                modified_matrix[i][j] = 0
                continue
            if matrix[i][j] == 1:
                continue
            weight = 0
            for k in range(j + 1, n):
                if matrix[i][k] == 1:
                    distance = k - j
                    weight = 0.5**distance
                    if weight < 0.1:
                        weight = 0.1
                    break
            modified_matrix[i][j] = weight if weight > 0 else matrix[i][j]

    return modified_matrix


# Process files in batches
i = 1
j = dim
for ins in range(1, num_samples + 1):
    if j <= num_samples:
        for x in range(i, j + 1):
            file_path = f"data/generated/instance_{x}.json"
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                modified_connections = modify_connections(data["componentConnections"])

                microservices.append(data)
                microservices_edges.append(modified_connections)
                computingnodes.append(data["computingNodes"])
                computingnodes_edges.append(data["infraConnections"])

                # print(f"Instance {x} modified matrix:", modified_connections)
            else:
                print(f"Warning: File {file_path} not found.")

        # Debugging: Print batch size
        print(f"Batch {i} to {j}: {len(microservices)} instances collected")

        # Call GNN model for the batch
        if microservices:
            (
                microservices_embedding,
                microservices_edges_embedding,
                computingnodes_embedding,
                computingnodes_edges_embedding,
                updated_parameters,
            ) = gnn_model(
                microservices,
                microservices_edges,
                computingnodes,
                computingnodes_edges,
                parameters,
                device=device,
            )
            # Debugging: Print embedding shapes
            print(
                f"Batch {i} to {j}: computingnodes_embedding: {len(computingnodes_embedding)} instances"
            )
            print(
                f"Batch {i} to {j}: microservices_embedding: {len(microservices_embedding)} instances"
            )
            if computingnodes_embedding:
                print(
                    f"Batch {i} to {j}: computingnodes_embedding[0]: {len(computingnodes_embedding[0])} nodes"
                )
            if computingnodes_edges:
                print(
                    f"Batch {i} to {j}: computingnodes_edges[0] shape: {np.array(computingnodes_edges[0]).shape}"
                )

        # Call AR model for the batch
        if computingnodes:
            (placements, updated_parameters) = AR_model(
                microservices,
                microservices_edges,
                computingnodes,
                computingnodes_edges,
                microservices_embedding,
                microservices_edges_embedding,
                computingnodes_embedding,
                computingnodes_edges_embedding,
                parameters,
                device=device,
            )
            # Process placements (list of placement dictionaries)
            for idx, placement in enumerate(placements):
                print(f"Placement for instance {i + idx}: {placement}")
            # Update parameters for next iteration
            parameters = updated_parameters
            # Clear lists for next batch
            microservices.clear()
            microservices_edges.clear()
            computingnodes.clear()
            computingnodes_edges.clear()
        i = j + 1
        j = j + dim
    else:
        for x in range(i, num_samples + 1):
            file_path = f"data/generated/instance_{x}.json"
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                modified_connections = modify_connections(data["componentConnections"])

                microservices.append(data)
                microservices_edges.append(modified_connections)
                computingnodes.append(data["computingNodes"])
                computingnodes_edges.append(data["infraConnections"])

                # print(f"Instance {x} modified matrix:", modified_connections)
            else:
                print(f"Warning: File {file_path} not found.")

        # Debugging: Print batch size
        print(
            f"Final batch {i} to {num_samples}: {len(microservices)} instances collected"
        )

        # Call GNN model for the final batch
        if microservices:
            (
                microservices_embedding,
                microservices_edges_embedding,
                computingnodes_embedding,
                computingnodes_edges_embedding,
                updated_parameters,
            ) = gnn_model(
                microservices,
                microservices_edges,
                computingnodes,
                computingnodes_edges,
                parameters,
                device=device,
            )
            print(f"Final batch {i} to {num_samples} embeddings computed.")
            # Debugging: Print embedding shapes
            print(
                f"Final batch {i} to {num_samples}: computingnodes_embedding: {len(computingnodes_embedding)} instances"
            )
            print(
                f"Final batch {i} to {num_samples}: microservices_embedding: {len(microservices_embedding)} instances"
            )
            if computingnodes_embedding:
                print(
                    f"Final batch {i} to {num_samples}: computingnodes_embedding[0]: {len(computingnodes_embedding[0])} nodes"
                )
            if computingnodes_edges:
                print(
                    f"Final batch {i} to {num_samples}: computingnodes_edges[0] shape: {np.array(computingnodes_edges[0]).shape}"
                )

            # Call AR model for the final batch
            if computingnodes:
                (placements, updated_parameters) = AR_model(
                    microservices,
                    microservices_edges,
                    computingnodes,
                    computingnodes_edges,
                    microservices_embedding,
                    microservices_edges_embedding,
                    computingnodes_embedding,
                    computingnodes_edges_embedding,
                    parameters,
                    device=device,
                )
                # Process placements (list of placement dictionaries)
                for idx, placement in enumerate(placements):
                    print(f"Placement for instance {i + idx}: {placement}")
                # Update parameters
                parameters = updated_parameters

            # Clear lists
            microservices.clear()
            microservices_edges.clear()
            computingnodes.clear()
            computingnodes_edges.clear()

print(f"Processed {num_samples} instances successfully.")
print(f"Device used: {device}, Hidden dimension: {dim}")
