import torch
import json
import os
import yaml
import copy
import numpy as np

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
charnum_service = config["model"]["charnum_service"]
charnum_s = config["model"]["charnum_s"]
charnum_n = config["model"]["charnum_n"]
charnum_se = config["model"]["charnum_se"]
charnum_ne = config["model"]["charnum_ne"]
charnum_node = config["model"]["charnum_node"]
charnum_component = config["model"]["charnum_component"]

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

def AR_model(
    services,
    dag,
    nodes,
    connectivity,
    emc,
    Cedge,
    ems,
    Sedge,
    parameters,
    device="cpu",
    settings_path="configs/setting.json",
):
    """
    AR model to assign service components to nodes for a batch of instances.

    Args:
        services (list): List of [batch_size] microservice data.
        dag (list): List of [batch_size] componentConnections matrices.
        nodes (list): List of [batch_size] computing nodes data.
        connectivity (list): List of [batch_size] infraConnections matrices.
        emc (list): List of [batch_size] microservices embeddings.
        Cedge (list): List of [batch_size] microservices edges embeddings.
        ems (list): List of [batch_size] computing nodes embeddings.
        Sedge (list): List of [batch_size] computing nodes edges embeddings.
        parameters (dict): Parameters (WSQ, WNK).
        device (str): Device to run computations ('cuda' or 'cpu').
        settings_path (str): Path to save settings.json.

    Returns:
        tuple: (placements, updated_parameters), where placements is a list of [batch_size] placement dictionaries.
    """
    # Move parameters to device
    parameters = {k: v.to(device) for k, v in parameters.items()}
    WSQ, WNK = parameters["WSQ"], parameters["WNK"]

    batch_size = len(services)
    placements = []

    for idx in range(batch_size):
        placement = {}

        # Debug: Print ems[idx] structure
        print(f"Instance {idx}: ems[idx] type: {type(ems[idx])}, length: {len(ems[idx])}, sample: {list(ems[idx].items())[:2] if isinstance(ems[idx], dict) else ems[idx][:2]}")

        # Prepare node embeddings (KS)
        node_embeddings = []
        try:
            # Handle ems[idx] as a dictionary or list
            if isinstance(ems[idx], dict):
                node_ids = list(ems[idx].keys())[:charnum_node]
                items = list(ems[idx].values())[:charnum_node]
            else:
                node_ids = [str(i) for i in range(charnum_node)]
                items = ems[idx][:charnum_node]
            for node_id, item in zip(node_ids, items):
                # Find corresponding node data
                node = next((n for n in nodes[idx] if str(n["nodeID"]) == node_id), None)
                if not node:
                    print(f"Warning: Instance {idx}: Node ID {node_id} not found in nodes")
                    continue
                # Map to characteristics regardless of item type
                embedding = [
                    node["characteristics"].get("cpu", 0),
                    node["characteristics"].get("memory", 0),
                    node["characteristics"].get("disk", 0),
                    node["characteristics"].get("other", 0),
                ]
                node_embeddings.append(embedding)
                # Debug: Print item shape if tensor/array
                if isinstance(item, (list, torch.Tensor, np.ndarray)):
                    item_shape = np.array(item).shape if isinstance(item, (list, np.ndarray)) else item.shape
                    print(f"Instance {idx}: Node {node_id} embedding shape from gnn_model: {item_shape}")
        except Exception as e:
            print(f"Error: Instance {idx}: Failed to process ems[idx]: {e}")
            placements.append(placement)
            continue

        if not node_embeddings:
            print(f"Warning: Instance {idx}: No valid node embeddings.")
            placements.append(placement)
            continue
        if len(node_embeddings) != charnum_node:
            print(f"Warning: Instance {idx}: Expected {charnum_node} node embeddings, got {len(node_embeddings)}")

        # Convert embeddings to tensor
        try:
            node_embeddings_tensor = torch.tensor(
                node_embeddings, dtype=torch.float32, device=device
            )
            if node_embeddings_tensor.ndim > 2:
                node_embeddings_tensor = node_embeddings_tensor.squeeze()
            if node_embeddings_tensor.ndim == 1:
                node_embeddings_tensor = node_embeddings_tensor.unsqueeze(0)
            print(f"Instance {idx}: node_embeddings_tensor shape: {node_embeddings_tensor.shape}")
        except Exception as e:
            print(f"Error: Instance {idx}: Failed to convert embeddings to tensor: {e}")
            placements.append(placement)
            continue

        # Validate WNK
        if WNK.ndim > 2:
            WNK = WNK.squeeze()
        if WNK.ndim != 2 or WNK.shape[0] != charnum_n:
            print(f"Error: Instance {idx}: WNK shape {WNK.shape} incompatible with expected input dim {charnum_n}")
            placements.append(placement)
            continue
        if node_embeddings_tensor.shape[1] != charnum_n:
            print(f"Error: Instance {idx}: node_embeddings_tensor shape {node_embeddings_tensor.shape} incompatible with charnum_n {charnum_n}")
            placements.append(placement)
            continue

        # Compute KS
        KS = node_embeddings_tensor @ WNK  # Shape: (charnum_node, dim)
        KS = (KS - torch.min(KS, dim=1, keepdim=True)[0]) / (
            torch.max(KS, dim=1, keepdim=True)[0]
            - torch.min(KS, dim=1, keepdim=True)[0]
            + 1e-10
        )
        print(f"Instance {idx}: KS shape: {KS.shape}")

        # Convert connectivity to tensor
        if isinstance(connectivity[idx], list):
            connectivity_array = np.array(connectivity[idx])
        else:
            connectivity_array = connectivity[idx]
        if connectivity_array.ndim != 3:
            print(f"Error: Instance {idx}: connectivity must be 3D, got shape {connectivity_array.shape}")
            placements.append(placement)
            continue
        if connectivity_array.shape[0] < charnum_node or connectivity_array.shape[1] < charnum_node:
            print(f"Error: Instance {idx}: connectivity shape {connectivity_array.shape} too small for {charnum_node} nodes")
            placements.append(placement)
            continue
        connectivity_matrix = torch.tensor(
            connectivity_array[:charnum_node, :charnum_node, 0], dtype=torch.float32, device=device
        )
        print(f"Instance {idx}: Connectivity matrix shape: {connectivity_matrix.shape}")

        # Store initial node resources
        initial_resources = {
            str(node["nodeID"]): {
                "cpu": node["characteristics"]["cpu"],
                "memory": node["characteristics"]["memory"],
                "disk": node["characteristics"]["disk"],
            }
            for node in nodes[idx][:charnum_node]
        }
        S = copy.deepcopy(initial_resources)
        print(f"Instance {idx}: Number of nodes in resources: {len(initial_resources)}")

        # Collect all components across services
        all_components = []
        component_specs = []
        component_ids = []
        for z, (service_id, service_components) in enumerate(emc[idx].items(), 1):
            print(f"Instance {idx}: Processing Service {z}: {service_id}")
            service_number = service_id.replace("Service", "")
            service = next(
                (
                    s
                    for s in services[idx].get("services", [])
                    if str(s.get("serviceID", "")) == service_number
                ),
                None,
            )
            if not service:
                print(f"Warning: Instance {idx}: No service found for ID {service_number}, skipping.")
                continue
            for i, hc_i in enumerate(service_components, 1):
                component_id = f"S{service_number}_C{i}"
                all_components.append(hc_i)
                component_specs.append(service["components"][i - 1]["characteristics"])
                component_ids.append(component_id)

        if not all_components:
            print(f"Warning: Instance {idx}: No components to process.")
            placements.append(placement)
            continue
        print(
            f"Instance {idx}: Expected total components: {charnum_service * charnum_component}, Actual: {len(all_components)}"
        )

        # Compute qc for components
        qc = torch.tensor(all_components, dtype=torch.float32, device=device) @ WSQ
        qc = (qc - torch.min(qc, dim=1, keepdim=True)[0]) / (
            torch.max(qc, dim=1, keepdim=True)[0]
            - torch.min(qc, dim=1, keepdim=True)[0]
            + 1e-10
        )
        print(f"Instance {idx}: qc shape: {qc.shape}, KS shape: {KS.shape}")

        # Compute compatibility scores
        u = qc @ KS.mT

        # Adjust scores with connectivity
        connectivity_factor = torch.mean(connectivity_matrix, dim=1)
        u *= (1 + connectivity_factor).unsqueeze(0)

        # Check resource constraints
        resource_mask = torch.ones_like(u, device=device)
        for j in range(charnum_node):
            node_key = str(nodes[idx][j]["nodeID"])
            for i, specs in enumerate(component_specs):
                if (
                    S[node_key]["cpu"] < specs["cpu"]
                    or S[node_key]["memory"] < specs["memory"]
                    or S[node_key]["disk"] < specs["disk"]
                ):
                    resource_mask[i, j] = float("-inf")

        u = u + resource_mask

        # Compute probabilities
        finite_u = torch.where(torch.isinf(u), torch.tensor(-1e10, device=device), u)
        shifted_u = finite_u - torch.max(finite_u, dim=1, keepdim=True)[0]
        eu = torch.exp(shifted_u)
        zigma_u = torch.sum(eu, dim=1, keepdim=True)
        p = eu / (zigma_u + 1e-10)

        # Handle invalid components
        invalid_components = (zigma_u.squeeze() == 0).nonzero(as_tuple=True)[0]
        selected_nodes = torch.argmax(p, dim=1)

        for idx_invalid in invalid_components:
            print(f"Warning: Instance {idx}: No nodes available for {component_ids[idx_invalid]}, using resource scores")
            resource_scores = [
                max(S[str(nodes[idx][j]["nodeID"])]["cpu"] - component_specs[idx_invalid]["cpu"], 0)
                + max(S[str(nodes[idx][j]["nodeID"])]["memory"] - component_specs[idx_invalid]["memory"], 0)
                + max(S[str(nodes[idx][j]["nodeID"])]["disk"] - component_specs[idx_invalid]["disk"], 0)
                for j in range(charnum_node)
            ]
            selected_nodes[idx_invalid] = torch.argmax(torch.tensor(resource_scores, device=device))

        # Update resources and record placements
        for i, component_id in enumerate(component_ids):
            selected_node_idx = selected_nodes[i].item()
            selected_node = str(nodes[idx][selected_node_idx]["nodeID"])
            specs = component_specs[i]
            S[selected_node]["cpu"] -= specs["cpu"]
            S[selected_node]["memory"] -= specs["memory"]
            S[selected_node]["disk"] -= specs["disk"]
            placement[component_id] = selected_node
            print(f"Instance {idx}: {component_id} assigned to Node {selected_node}")

        print(f"Instance {idx}: Resources after placement:")
        for n_key, res in S.items():
            print(f"  {n_key}: cpu={res['cpu']:.2f}, memory={res['memory']:.2f}, disk={res['disk']:.2f}")

        placements.append(placement)

    # Save parameters
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    with open(settings_path, "w") as f:
        formatted_parameters = {
            key: [[round(float(val), 2) for val in row] for row in param.cpu().tolist()]
            for key, param in parameters.items()
        }
        json.dump(formatted_parameters, f, indent=4)

    return placements, parameters
