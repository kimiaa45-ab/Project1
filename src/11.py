import json
import os
import torch
from typing import List, Tuple


def sigmoid(x):
    return torch.sigmoid(x)


def relu(x):
    return torch.relu(x)


def normalize_vector(v, min_val=0.1, max_val=1.0):
    v_min, v_max = torch.min(v), torch.max(v)
    if v_max == v_min:
        return torch.full_like(v, min_val)
    return min_val + (max_val - min_val) * (v - v_min) / (v_max - v_min)


def msg_component_processing(input_pairs: List[Tuple[str, str]], output_prefixes: List[str]):
    """
    Process multiple input file pairs and generate corresponding outputs.

    Args:
        input_pairs: List of tuples, each containing (services_file, connections_file)
        output_prefixes: List of prefixes for output files (e.g., ['output1', 'output2'])
    """
    if len(input_pairs) != len(output_prefixes):
        raise ValueError("Number of input pairs must match number of output prefixes")

    for idx, ((services_file, connections_file), output_prefix) in enumerate(zip(input_pairs, output_prefixes)):
        print(f"\nProcessing input pair {idx + 1}/{len(input_pairs)}")
        print(f"Step 1: Reading and normalizing {services_file}")

        # Read services file
        with open(services_file, "r", encoding="utf-8") as f:
            component_data = json.load(f)
        print(f"  - Number of services: {len(component_data)}")

        app_h_C_final = {}
        app_e_C_final = {}

        for service in component_data:
            service_name = f"Service{service['serviceID']}"
            components = service["components"]
            user_id = service["userID"]
            helper_id = service["helperID"]
            print(f"\nProcessing Service: {service_name} (User: {user_id}, Helper: {helper_id})")
            num_components = len(components)
            print(f"  - Number of components: {num_components}")

            # Build feature vector
            component_vectors = torch.tensor([[c["cpu"], c["memory"], c["dataSize"], c["disk"], c["reliabilityScore"]]
                                              for c in components], dtype=torch.float32)
            norm_components = torch.stack([normalize_vector(component_vectors[:, i])
                                           for i in range(component_vectors.shape[1])]).T

            # Compute h_C_0
            WC = torch.rand(5, 128) * 2 - 1
            h_C_layers = [norm_components @ WC]

            # Load dependencies
            with open(connections_file, "r", encoding="utf-8") as f:
                dependency_data = json.load(f)
            app_deps = dependency_data.get(service_name, {})
            dep_values = torch.tensor([val for val in app_deps.values() if val > 0],
                                      dtype=torch.float32)
            norm_deps = normalize_vector(dep_values) if len(dep_values) > 0 else torch.zeros(1)

            WEC = torch.rand(128) * 2 - 1
            e_C_dict = {}
            idx_dep = 0
            for edge_key, val in app_deps.items():
                if val > 0:
                    e_C_dict[edge_key] = norm_deps[idx_dep] * WEC
                    idx_dep += 1
            sigmoid_e_C = {key: sigmoid(val) for key, val in e_C_dict.items()}

            MC = torch.rand(128, 128) * 2 - 1
            NC = torch.rand(128, 128) * 2 - 1

            # Component messaging
            for layer in range(num_components - 1):
                h_C_current = h_C_layers[layer]
                MC_h_C = h_C_current @ MC
                NC_h_C = h_C_current @ NC
                h_C_next = h_C_current.clone()

                for i in range(num_components):
                    neighbors = set()
                    for edge_key in e_C_dict.keys():
                        c1, c2 = map(int, edge_key.split('_c')[1:])
                        if c1 == i + 1:
                            neighbors.add(c2 - 1)
                        elif c2 == i + 1:
                            neighbors.add(c1 - 1)
                    neighbors = list(neighbors)

                    neighbor_sum = torch.zeros(128)
                    for j in neighbors:
                        edge_key = f"e_c{i + 1}_c{j + 1}" if i < j else f"e_c{j + 1}_c{i + 1}"
                        if edge_key in sigmoid_e_C:
                            sig_e_ij = sigmoid_e_C[edge_key]
                            nc_h_j = NC_h_C[j]
                            neighbor_sum += sig_e_ij * nc_h_j

                    mc_h_i = MC_h_C[i]
                    aggr_result = mc_h_i + neighbor_sum
                    norm_aggr = normalize_vector(aggr_result)
                    relu_result = relu(norm_aggr)
                    h_C_next[i] = h_C_current[i] + relu_result

                h_C_layers.append(h_C_next)

            # Edge messaging
            X = torch.rand(128, 128) * 2 - 1
            Y = torch.rand(128, 128) * 2 - 1
            Z = torch.rand(128, 128) * 2 - 1
            e_C_layers = [e_C_dict]

            for layer in range(num_components - 1):
                e_C_current = e_C_layers[layer]
                h_C_current = h_C_layers[layer]
                e_C_next = e_C_current.copy()
                for edge_key in e_C_current.keys():
                    c1, c2 = map(int, edge_key.split('_c')[1:])
                    i, j = c1 - 1, c2 - 1
                    e_ij = e_C_current[edge_key]
                    h_i, h_j = h_C_current[i], h_C_current[j]
                    X_e_ij = e_ij @ X
                    Y_h_i = h_i @ Y
                    Z_h_j = h_j @ Z
                    aggr = X_e_ij + Y_h_i + Z_h_j
                    norm_aggr = normalize_vector(aggr)
                    relu_result = relu(norm_aggr)
                    e_C_next[edge_key] = e_ij + relu_result
                e_C_layers.append(e_C_next)

            # Store final layer
            final_h_C_rounded = torch.round(h_C_layers[-1] * 100) / 100
            final_e_C_rounded = {key: torch.round(val * 100) / 100
                                 for key, val in e_C_layers[-1].items()}

            app_h_C_final[service_name] = final_h_C_rounded.tolist()
            app_e_C_final[service_name] = {key: val.tolist()
                                           for key, val in final_e_C_rounded.items()}

            # Display final layer results
            print(f"\nFinal Layer Results for {service_name} (User: {user_id}, Helper: {helper_id}):")
            for i, h in enumerate(app_h_C_final[service_name]):
                formatted_h = [f"{x:.2f}" for x in h[:5]]
                print(f"  h_C_final[Component {i + 1}]: {formatted_h}...")
            for edge_key, e in list(app_e_C_final[service_name].items())[:10]:
                formatted_h = [f"{x:.2f}" for x in e[:5]]
                print(f"  e_C_final[{edge_key}]: {formatted_h}...")

        # Save results with unique filenames
        h_output_file = f"{output_prefix}_component.json"
        e_output_file = f"{output_prefix}_Cedge.json"
        with open(h_output_file, "w", encoding="utf-8") as f:
            json.dump(app_h_C_final, f, indent=4)
        with open(e_output_file, "w", encoding="utf-8") as f:
            json.dump(app_e_C_final, f, indent=4)
        print(f"\nFinal embeddings saved to {h_output_file} and {e_output_file}")


if __name__ == "__main__":
    # Example: Define multiple input file pairs and output prefixes
    input_pairs = [
        ("services1.json", "newcomponentsConnections1.json"),
        ("services2.json", "newcomponentsConnections2.json"),
        # Add more pairs as needed
    ]
    output_prefixes = ["msg1", "msg2"]  # Unique prefixes for output files

    msg_component_processing(input_pairs, output_prefixes)