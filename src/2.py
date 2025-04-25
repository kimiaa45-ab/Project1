import json
import os
import torch


def sigmoid(x):
    return torch.sigmoid(x)


def relu(x):
    return torch.relu(x)


def normalize_vector(v, min_val=0.1, max_val=1.0):
    v_min, v_max = torch.min(v), torch.max(v)
    if v_max == v_min or torch.isnan(v_max) or torch.isnan(v_min):
        return torch.full_like(v, min_val)
    return min_val + (max_val - min_val) * (v - v_min) / (v_max - v_min)


def msg_node_processing(sample_index=0, inner_sample_index=0):
    print(f"Step 1: Reading and extracting data for sample index {sample_index} from sample_1.json")

    # خواندن فایل sample_1.json
    try:
        with open("../data/generated/sample_1.json", "r", encoding="utf-8") as f:
            sample_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("File ../data/generated/sample_1.json not found. Please check the path.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in sample_1.json: {e}")

    # چاپ نوع و نمونه‌ای از sample_data برای دیباگ
    print(f"Type of sample_data: {type(sample_data)}")
    print(f"First few items in sample_data: {sample_data[:3] if isinstance(sample_data, list) else sample_data}")

    # اگر sample_data رشته باشد، به دیکشنری تبدیلش می‌کنیم
    if isinstance(sample_data, str):
        try:
            sample_data = json.loads(sample_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"sample_data is a string but not valid JSON: {e}")

    # اگر sample_data لیست نیست، به لیست تبدیلش می‌کنیم
    if not isinstance(sample_data, list):
        sample_data = [sample_data]

    # تبدیل رشته‌های JSON به دیکشنری
    parsed_samples = []
    for sample in sample_data:
        if isinstance(sample, str):
            try:
                parsed_samples.append(json.loads(sample))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON string: {sample[:100]}... Error: {e}")
                continue
        elif isinstance(sample, dict):
            parsed_samples.append(sample)
        else:
            print(f"Warning: Skipping invalid sample type: {type(sample)}, value: {str(sample)[:100]}...")
            continue

    # چاپ تعداد نمونه‌های معتبر برای دیباگ
    print(f"Number of valid parsed samples: {len(parsed_samples)}")

    # انتخاب نمونه بر اساس اندیس
    if not parsed_samples:
        raise ValueError("No valid samples found in sample_1.json")
    if sample_index >= len(parsed_samples):
        raise ValueError(f"Sample index {sample_index} out of range. Only {len(parsed_samples)} samples available.")

    target_sample = parsed_samples[sample_index]
    print(f"Selected sample at index {sample_index}: {list(target_sample.keys())}")

    # اگر کلید "sample" وجود داره، داده‌ها رو از اونجا بخون
    if "sample" in target_sample:
        sample_content = target_sample["sample"]
        if isinstance(sample_content, list):
            print(f"target_sample['sample'] is a list with {len(sample_content)} items")
            if not sample_content:
                raise ValueError("target_sample['sample'] is an empty list")
            if inner_sample_index >= len(sample_content):
                raise ValueError(
                    f"Inner sample index {inner_sample_index} out of range. Only {len(sample_content)} inner samples available.")
            data_source = sample_content[inner_sample_index]
            print(
                f"Selected inner sample at index {inner_sample_index}: {list(data_source.keys()) if isinstance(data_source, dict) else data_source}")
        else:
            data_source = sample_content
            print(
                f"target_sample['sample'] is not a list, using it directly: {list(data_source.keys()) if isinstance(data_source, dict) else data_source}")
    else:
        data_source = target_sample
        print(
            f"No 'sample' key found, using target_sample directly: {list(data_source.keys()) if isinstance(data_source, dict) else data_source}")

    # مطمئن بشیم data_source یه دیکشنریه
    if not isinstance(data_source, dict):
        raise ValueError(f"data_source is not a dictionary: {type(data_source)}, value: {str(data_source)[:100]}...")

    # استخراج computingNodes
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

    # ساخت بردار ویژگی‌ها برای نودها
    node_vectors = torch.tensor([[n["characteristics"]["cpu"],
                                  n["characteristics"]["memory"],
                                  n["characteristics"]["disk"],
                                  n["characteristics"]["reliabilityScore"]]
                                 for n in node_data], dtype=torch.float32)
    norm_nodes = torch.stack([normalize_vector(node_vectors[:, i])
                              for i in range(node_vectors.shape[1])]).T

    # استخراج و تبدیل infraConnections
    dependency_data = {}
    for node in node_data:
        node_name = f"Node{node['nodeID']}"
        app_deps = {}
        connections = data_source.get("infraConnections", [])
        print(f"infraConnections for {node_name}: {connections[:100]}...")

        # چک کردن ساختار connections
        if connections and isinstance(connections, list):
            try:
                bandwidth = torch.zeros((num_nodes, num_nodes))
                delay = torch.zeros((num_nodes, num_nodes))
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
    num_layers = 4  # تعداد لایه‌ها برای پیام‌رسانی

    for node in node_data:
        node_name = f"Node{node['nodeID']}"
        print(f"\nProcessing Node: {node_name}")
        num_nodes = len(node_data)
        print(f"  - Number of nodes: {num_nodes}")

        # محاسبه h_N_0
        WN = torch.rand(4, 128) * 2 - 1
        h_N_layers = [norm_nodes @ WN]

        # استفاده از dependency_data
        app_deps = dependency_data.get(node_name, {})
        dep_values = torch.tensor([val for pair in app_deps.values() for val in pair if val > 0],
                                  dtype=torch.float32)
        norm_deps = normalize_vector(dep_values) if len(dep_values) > 0 else torch.zeros(1)

        WEN = torch.rand(2, 128) * 2 - 1
        e_N_dict = {}
        idx = 0
        for edge_key, val in app_deps.items():
            if val[0] > 0 or val[1] > 0:
                edge_features = torch.tensor([val[0], val[1]], dtype=torch.float32)
                e_N_dict[edge_key] = edge_features @ WEN
                idx += 1
        sigmoid_e_N = {key: sigmoid(val) for key, val in e_N_dict.items()}

        MN = torch.rand(128, 128) * 2 - 1
        NN = torch.rand(128, 128) * 2 - 1

        # پیام‌رسانی نودها
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

                neighbor_sum = torch.zeros(128)
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

        # پیام‌رسانی لبه‌ها
        X = torch.rand(128, 128) * 2 - 1
        Y = torch.rand(128, 128) * 2 - 1
        Z = torch.rand(128, 128) * 2 - 1
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

        # ذخیره لایه آخر
        final_h_N_rounded = torch.round(h_N_layers[-1] * 100) / 100
        final_e_N_rounded = {key: torch.round(val * 100) / 100
                             for key, val in e_N_layers[-1].items()}

        app_h_N_final[node_name] = final_h_N_rounded.tolist()
        app_e_N_final[node_name] = {key: val.tolist()
                                    for key, val in final_e_N_rounded.items()}

        # نمایش لایه آخر
        print(f"\nFinal Layer Results for {node_name}:")
        for i, h in enumerate(app_h_N_final[node_name]):
            formatted_h = [f"{x:.2f}" for x in h[:5]]
            print(f"  h_N_final[Node {i + 1}]: {formatted_h}...")
        for edge_key, e in list(app_e_N_final[node_name].items())[:10]:
            formatted_e = [f"{x:.2f}" for x in e[:5]]
            print(f"  e_N_final[{edge_key}]: {formatted_e}...")

    # ذخیره نتایج
    with open("../data/processed/msg_node1.json", "w", encoding="utf-8") as f:
        json.dump(app_h_N_final, f, indent=4)
    with open("../data/processed/msg_Nedge1.json", "w", encoding="utf-8") as f:
        json.dump(app_e_N_final, f, indent=4)
    print("\nFinal embeddings saved for all nodes")


if __name__ == "__main__":
    msg_node_processing(sample_index=0, inner_sample_index=0)