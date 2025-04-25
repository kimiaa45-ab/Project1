import torch
import json


def sigmoid(x):
    return torch.sigmoid(x)


def relu(x):
    return torch.relu(x)


def normalize_vector(v, min_val=0.1, max_val=1.0):
    v_min, v_max = torch.min(v), torch.max(v)
    if v_max == v_min or torch.isnan(v_max) or torch.isnan(v_min):
        return torch.full_like(v, min_val)
    return min_val + (max_val - min_val) * (v - v_min) / (v_max - v_min)


def msg_node_processing(sample, device='cpu'):
    """
    Process a single sample to compute node GNN embeddings.

    Args:
        sample (dict): Input sample dictionary containing computingNodes and infraConnections.
        device (str): Device to run computations ('cuda' or 'cpu').

    Returns:
        tuple: (app_h_N_final, app_e_N_final) containing node and edge embeddings.
    """
    torch_device = torch.device(device)
    print(f"Processing sample with keys: {list(sample.keys())}")

    # Extract data_source directly from sample
    if "sample" in sample:
        sample_content = sample["sample"]
        if isinstance(sample_content, list):
            if not sample_content:
                raise ValueError("sample['sample'] is an empty list")
            data_source = sample_content[0]  # Take first inner sample
            print(
                f"Selected inner sample: {list(data_source.keys()) if isinstance(data_source, dict) else data_source}")
        else:
            data_source = sample_content
            print(
                f"sample['sample'] is not a list, using directly: {list(data_source.keys()) if isinstance(data_source, dict) else data_source}")
    else:
        data_source = sample
        print(
            f"No 'sample' key found, using sample directly: {list(data_source.keys()) if isinstance(data_source, dict) else data_source}")

    # Ensure data_source is a dictionary
    if not isinstance(data_source, dict):
        raise ValueError(f"data_source is not a dictionary: {type(data_source)}, value: {str(data_source)[:100]}...")

    # Extract computingNodes
    node_data = []
    for node in data_source.get("computingNodes", []):
        node_info = {
            "nodeID": node.get("nodeID", 0),
            "characteristics": {
                "cpu": node["characteristics"].get("cpu", 0),
                "memory": node["characteristics"].get("memory", 0),
                "disk": node["characteristics"].get("disk", 0),
                "reliabilityScore": node["characteristics"].get("reliabilityScore", 0)
            }
        }
        node_data.append(node_info)

    num_nodes = len(node_data)
    print(f"  - Number of computing nodes: {num_nodes}")

    # Build feature vectors for nodes
    node_vectors = torch.tensor([[n["characteristics"]["cpu"],
                                  n["characteristics"]["memory"],
                                  n["characteristics"]["disk"],
                                  n["characteristics"]["reliabilityScore"]
                                 for n in node_data], dtype=torch.float32).to(torch_device)
    norm_nodes = torch.stack([normalize_vector(node_vectors[:, i])
                              for i in range(node_vectors.shape[1])]).T

    # Extract and convert infraConnections
    dependency_data = {}
    for node in node_data:
        node_name = f"Node{node['nodeID']}"
    app_deps = {}
    connections = data_source.get("infraConnections", [])
    print(f"infraConnections for {node_name}: {connections[:100]}...")

    if connections and isinstance(connections, list):
        try:
            bandwidth = torch.zeros((num_nodes, num_nodes), device=torch_device)
            delay = torch.zeros((num_nodes, num_nodes), device=torch_device)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if isinstance(connections[i][j], list) and len(connections[i][j]) == 2:
                        bandwidth[i][j] = connections[i][j][0]
                        delay[i][j] = connections[i][j][1]
                    else:
                        bandwidth[i][j] = float('-inf')
                        delay[i][j] = float('inf')

            finite_bandwidth = bandwidth[torch.isfinite(bandwidth)]
            finite_delay = delay[torch.isfinite(delay)]
            if len(finite_bandwidth) > 0 and len(finite_delay) > 0:
                norm_bandwidth = bandwidth.clone()
                norm_delay = delay.clone()
                norm_bandwidth[torch.isfinite(bandwidth)] = normalize_vector(finite_bandwidth)
                norm_delay[torch.isfinite(delay)] = normalize_vector(finite_delay)
            else:
                norm_bandwidth = torch.full_like(bandwidth, 0.1)
                norm_delay = torch.full_like(delay, 0.1)

            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if bandwidth[i][j] != float('-inf'):
                        edge_key = f"e_n{i + 1}_n{j + 1}"
                        app_deps[edge_key] = [norm_bandwidth[i][j], norm_delay[i][j]]
        except (TypeError, IndexError) as e:
            print(f"Warning: Invalid infraConnections format for {node_name}: {e}. Skipping dependency extraction.")
            app_deps = {}
        else:
            print(f"Warning: infraConnections is not a valid matrix for {node_name}. Skipping dependency extraction.")
            app_deps = {}

    dependency_data[node_name] = app_deps

    app_h_N_final = {}
    app_e_N_final = {}
    num_layers = 4  # Fixed number of layers

    for node in node_data:
        node_name = f"Node{node['nodeID']}"
        print(f"\nProcessing Node: {node_name}")
        print(f"  - Number of nodes: {num_nodes}")

        # Compute h_N_0
        WN = torch.rand(4, 128, device=torch_device) * 2 - 1
        h_N_layers = [norm_nodes @ WN]

        # Use dependency_data
        app_deps = dependency_data.get(node_name, {})
        dep_values = torch.tensor([val for pair in app_deps.values() for val in pair if val > 0],
                                  dtype=torch.float32, device=torch_device)
        norm_deps = normalize_vector(dep_values) if len(dep_values) > 0 else torch.zeros(1, device=torch_device)

        WEN = torch.rand(2, 128, device=torch_device) * 2 - 1
        e_N_dict = {}
        idx = 0
        for edge_key, val in app_deps.items():
            if val[0] > 0 or val[1] > 0:
                edge_features = torch.tensor([val[0], val[1]], dtype=torch.float32, device=torch_device)
                e_N_dict[edge_key] = edge_features @ WEN
                idx += 1
        sigmoid_e_N = {key: sigmoid(val) for key, val in e_N_dict.items()}

        MN = torch.rand(128, 128, device=torch_device) * 2 - 1
        NN = torch.rand(128, 128, device=torch_device) * 2 - 1

        # Node message passing
        for layer in range(num_layers):
            h_N_current = h_N_layers[layer]
            MN_h_N = h_N_current @ MN
            NN_h_N = h_N_current @ NN
            h_N_next = h_N_current.clone()

            for i in range(num_nodes):
                neighbors = set()
                for edge_key in e_N_dict.keys():
                    n1, n2 = map(int, edge_key.split('_n')[1:])
                    if n1 == i + 1:
                        neighbors.add(n2 - 1)
                    elif n2 == i + 1:
                        neighbors.add(n1 - 1)
                neighbors = list(neighbors)

                neighbor_sum = torch.zeros(128, device=torch_device)
                for j in neighbors:
                    edge_key = f"e_n{i + 1}_n{j + 1}" if i < j else f"e_n{j + 1}_n{i + 1}"
                    if edge_key in sigmoid_e_N:
                        sig_e_ij = sigmoid_e_N[edge_key]
                        nn_h_j = NN_h_N[j]
                        neighbor_sum += sig_e_ij * nn_h_j

                mn_h_i = MN_h_N[i]
                aggr_result = mn_h_i + neighbor_sum
                norm_aggr = normalize_vector(aggr_result)
                relu_result = relu(norm_aggr)
                h_N_next[i] = h_N_current[i] + relu_result

            h_N_layers.append(h_N_next)

        # Edge message passing
        X = torch.rand(128, 128, device=torch_device) * 2 - 1
        Y = torch.rand(128, 128, device=torch_device) * 2 - 1
        Z = torch.rand(128, 128, device=torch_device) * 2 - 1
        e_N_layers = [e_N_dict]

        for layer in range(num_layers):
            e_N_current = e_N_layers[layer]
            h_N_current = h_N_layers[layer]
            e_N_next = e_N_current.copy()
            for edge_key in e_N_current.keys():
                n1, n2 = map(int, edge_key.split('_n')[1:])
                i, j = n1 - 1, n2 - 1
                e_ij = e_N_current[edge_key]
                h_i, h_j = h_N_current[i], h_N_current[j]
                X_e_ij = e_ij @ X
                Y_h_i = h_i @ Y
                Z_h_j = h_j @ Z
                aggr = X_e_ij + Y_h_i + Z_h_j
                norm_aggr = normalize_vector(aggr)
                relu_result = relu(norm_aggr)
                e_N_next[edge_key] = e_ij + relu_result
            e_N_layers.append(e_N_next)

        # Store final layer
        final_h_N_rounded = torch.round(h_N_layers[-1] * 100) / 100
        final_e_N_rounded = {key: torch.round(val * 100) / 100
                             for key, val in e_N_layers[-1].items()}

        app_h_N_final[node_name] = final_h_N_rounded.tolist()
        app_e_N_final[node_name] = {key: val.tolist()
                                    for key, val in final_e_N_rounded.items()}

        # Display final layer
        print(f"\nFinal Layer Results for {node_name}:")
        for i, h in enumerate(app_h_N_final[node_name]):
            formatted_h = [f"{x:.2f}" for x in h[:5]]
            print(f"  h_N_final[Node {i + 1}]: {formatted_h}...")
        for edge_key, e in list(app_e_N_final[node_name].items())[:10]:
            formatted_e = [f"{x:.2f}" for x in e[:5]]
            print(f"  e_N_final[{edge_key}]: {formatted_e}...")

    print("\nFinal embeddings computed for all nodes")
    return app_h_N_final, app_e_N_final