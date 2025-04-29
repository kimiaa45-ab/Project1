import torch
import json
import os
import yaml
import copy

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
charnum_s = config["model"]["charnum_s"]
charnum_n = config["model"]["charnum_n"]
charnum_se = config["model"]["charnum_se"]
charnum_ne = config["model"]["charnum_ne"]

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
    AR model to assign service components to nodes in parallel using embeddings.

    Args:
        services (list): List of microservice data.
        dag (list): List of componentConnections matrices.
        nodes (list): List of computing nodes data.
        connectivity (list): List of infraConnections matrices.
        emc (list): Microservices embeddings.
        Cedge (list): Microservices edges embeddings.
        ems (list): Computing nodes embeddings.
        Sedge (list): Computing nodes edges embeddings.
        parameters (dict): Parameters (WSQ, WNK).
        device (str): Device to run computations ('cuda' or 'cpu').
        settings_path (str): Path to save settings.json.

    Returns:
        tuple: (placement, updated_parameters)
    """
    # Move parameters to device
    parameters = {k: v.to(device) for k, v in parameters.items()}
    WSQ, WNK = parameters["WSQ"], parameters["WNK"]

    # Initialize output
    placement = {}

    # Prepare node embeddings (KS) and normalize
    node_embeddings = []
    for node_dict in ems:
        for node_name, embedding in node_dict.items():
            node_embeddings.append(embedding)
    if not node_embeddings:
        print("Warning: No node embeddings provided.")
        return placement, parameters
    KS = (
        torch.tensor(node_embeddings, dtype=torch.float32, device=device) @ WNK
    )  # Shape: (num_nodes, 128)
    KS = (KS - torch.min(KS, dim=1, keepdim=True)[0]) / (
        torch.max(KS, dim=1, keepdim=True)[0]
        - torch.min(KS, dim=1, keepdim=True)[0]
        + 1e-10
    )
    num_nodes = KS.shape[0]

    # Convert connectivity to tensor
    connectivity_matrix = torch.tensor(
        connectivity[0], dtype=torch.float32, device=device
    )[
        :, :, 0
    ]  # Shape: (num_nodes, num_nodes)

    # Store initial node resources
    initial_resources = {
        str(node["nodeID"]): {
            "cpu": node["characteristics"]["cpu"],
            "memory": node["characteristics"]["memory"],
            "disk": node["characteristics"]["disk"],
        }
        for node in nodes[0]
    }
    S = copy.deepcopy(initial_resources)

    # Collect all components across services
    all_components = []
    component_specs = []
    component_ids = []
    for z, (service_id, service_components) in enumerate(emc[0].items(), 1):
        print(f"\nProcessing Service {z}: {service_id}")
        service_number = service_id.replace("Service", "")
        service = next(
            (
                s
                for s in services[0].get("services", [])
                if str(s.get("serviceID", "")) == service_number
            ),
            None,
        )
        if not service:
            print(f"Warning: No service found for ID {service_number}, skipping.")
            continue

        for i, hc_i in enumerate(service_components, 1):
            component_id = f"S{service_number}_C{i}"
            all_components.append(hc_i)
            component_specs.append(service["components"][i - 1]["characteristics"])
            component_ids.append(component_id)

    if not all_components:
        print("Warning: No components to process.")
        return placement, parameters

    # Compute qc for all components in parallel
    qc = (
        torch.tensor(all_components, dtype=torch.float32, device=device) @ WSQ
    )  # Shape: (num_components, 128)
    qc = (qc - torch.min(qc, dim=1, keepdim=True)[0]) / (
        torch.max(qc, dim=1, keepdim=True)[0]
        - torch.min(qc, dim=1, keepdim=True)[0]
        + 1e-10
    )

    # Compute compatibility scores (u) for all components and nodes
    u = qc @ KS.T  # Shape: (num_components, num_nodes)

    # Adjust scores with connectivity
    connectivity_factor = torch.mean(connectivity_matrix, dim=1)  # Shape: (num_nodes,)
    u *= (1 + connectivity_factor).unsqueeze(
        0
    )  # Broadcast: (num_components, num_nodes)

    # Check resource constraints in parallel
    resource_mask = torch.ones_like(
        u, device=device
    )  # Shape: (num_components, num_nodes)
    for j in range(num_nodes):
        node_key = str(nodes[0][j]["nodeID"])
        for i, specs in enumerate(component_specs):
            if (
                S[node_key]["cpu"] < specs["cpu"]
                or S[node_key]["memory"] < specs["memory"]
                or S[node_key]["disk"] < specs["disk"]
            ):
                resource_mask[i, j] = float("-inf")

    # Apply resource mask
    u = u + resource_mask  # Invalid assignments get -inf

    # Compute probabilities using stable softmax
    finite_u = torch.where(torch.isinf(u), torch.tensor(-1e10, device=device), u)
    shifted_u = finite_u - torch.max(finite_u, dim=1, keepdim=True)[0]
    eu = torch.exp(shifted_u)
    zigma_u = torch.sum(eu, dim=1, keepdim=True)
    p = eu / (zigma_u + 1e-10)  # Shape: (num_components, num_nodes)

    # Handle cases where no valid nodes are available
    invalid_components = (zigma_u.squeeze() == 0).nonzero(as_tuple=True)[0]
    selected_nodes = torch.argmax(p, dim=1)  # Shape: (num_components,)

    # Fallback for invalid components using resource scores
    for idx in invalid_components:
        print(
            f"Warning: No nodes available for {component_ids[idx]}, using resource scores"
        )
        resource_scores = [
            max(S[str(nodes[0][j]["nodeID"])]["cpu"] - component_specs[idx]["cpu"], 0)
            + max(
                S[str(nodes[0][j]["nodeID"])]["memory"]
                - component_specs[idx]["memory"],
                0,
            )
            + max(
                S[str(nodes[0][j]["nodeID"])]["disk"] - component_specs[idx]["disk"], 0
            )
            for j in range(num_nodes)
        ]
        selected_nodes[idx] = torch.argmax(torch.tensor(resource_scores, device=device))

    # Update resources and record placements
    for i, component_id in enumerate(component_ids):
        selected_node_idx = selected_nodes[i].item()
        selected_node = str(nodes[0][selected_node_idx]["nodeID"])
        specs = component_specs[i]
        S[selected_node]["cpu"] -= specs["cpu"]
        S[selected_node]["memory"] -= specs["memory"]
        S[selected_node]["disk"] -= specs["disk"]
        placement[component_id] = selected_node
        print(f"  - {component_id} assigned to Node {selected_node}")

    print(f"\nResources after placement:")
    for n_key, res in S.items():
        print(
            f"  {n_key}: cpu={res['cpu']:.2f}, memory={res['memory']:.2f}, disk={res['disk']:.2f}"
        )

    # Save parameters to settings.json
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    with open(settings_path, "w") as f:
        formatted_parameters = {
            key: [[round(float(val), 2) for val in row] for row in param.cpu().tolist()]
            for key, param in parameters.items()
        }
        json.dump(formatted_parameters, f, indent=4)

    updated_parameters = parameters

    return placement, updated_parameters
