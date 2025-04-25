import json
import os
import torch


def sigmoid(x):
    return torch.sigmoid(x)


def relu(x):
    return torch.relu(x)


def normalize_vector(v, min_val=0.1, max_val=1.0):
    v_min, v_max = torch.min(v), torch.max(v)
    if v_max == v_min:
        return torch.full_like(v, min_val)
    return min_val + (max_val - min_val) * (v - v_min) / (v_max - v_min)


def msg_component_processing(sample_index=0, inner_sample_index=0):
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

    # استخراج services
    component_data = []
    for service in data_source.get("services", []):
        service_data = {
            "serviceID": service.get("serviceID", 0),
            "components": [
                {
                    "cpu": comp["versions"][0]["characteristics"]["cpu"],
                    "memory": comp["versions"][0]["characteristics"]["memory"],
                    "dataSize": comp["versions"][0]["characteristics"]["dataSize"],
                    "disk": comp["versions"][0]["characteristics"]["disk"],
                    "reliabilityScore": comp["versions"][0]["characteristics"]["reliabilityScore"]
                } for comp in service.get("components", [])
            ],
            "userID": service.get("userID", data_source.get("usersNodes", [{}])[0].get("nodeID", 0)),
            "helperID": service.get("helperID", data_source.get("helperNodes", [{}])[0].get("nodeID", 0))
        }
        component_data.append(service_data)

    print(f"  - Number of services: {len(component_data)}")

    # استخراج و تبدیل componentConnections به دیکشنری
    dependency_data = {}
    for service in component_data:
        service_name = f"Service{service['serviceID']}"
        app_deps = {}
        connections = data_source.get("componentConnections", [])
        print(f"componentConnections for {service_name}: {connections[:100]}...")  # دیباگ

        # چک کردن ساختار connections
        if connections and isinstance(connections, list) and all(isinstance(row, list) for row in connections):
            # فرض می‌کنیم connections یه ماتریس دو بعدیه
            try:
                for i in range(len(connections)):
                    for j in range(len(connections[i])):
                        if connections[i][j] > 0:
                            edge_key = f"e_c{i + 1}_c{j + 1}" if i < j else f"e_c{j + 1}_c{i + 1}"
                            app_deps[edge_key] = connections[i][j]
            except (TypeError, IndexError) as e:
                print(
                    f"Warning: Invalid componentConnections format for {service_name}: {e}. Skipping dependency extraction.")
                app_deps = {}
        else:
            print(
                f"Warning: componentConnections is not a valid matrix for {service_name}. Skipping dependency extraction.")
            app_deps = {}

        dependency_data[service_name] = app_deps

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

        # ساخت بردار ویژگی‌ها
        component_vectors = torch.tensor([[c["cpu"], c["memory"], c["dataSize"], c["disk"], c["reliabilityScore"]]
                                          for c in components], dtype=torch.float32)
        norm_components = torch.stack([normalize_vector(component_vectors[:, i])
                                       for i in range(component_vectors.shape[1])]).T

        # محاسبه h_C_0
        WC = torch.rand(5, 128) * 2 - 1
        h_C_layers = [norm_components @ WC]

        # استفاده از dependency_data
        app_deps = dependency_data.get(service_name, {})
        dep_values = torch.tensor([val for val in app_deps.values() if val > 0],
                                  dtype=torch.float32)
        norm_deps = normalize_vector(dep_values) if len(dep_values) > 0 else torch.zeros(1)

        WEC = torch.rand(128) * 2 - 1
        e_C_dict = {}
        idx = 0
        for edge_key, val in app_deps.items():
            if val > 0:
                e_C_dict[edge_key] = norm_deps[idx] * WEC
                idx += 1
        sigmoid_e_C = {key: sigmoid(val) for key, val in e_C_dict.items()}

        MC = torch.rand(128, 128) * 2 - 1
        NC = torch.rand(128, 128) * 2 - 1

        # پیام‌رسانی کامپوننت‌ها
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

        # پیام‌رسانی لبه‌ها
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

        # ذخیره لایه آخر
        final_h_C_rounded = torch.round(h_C_layers[-1] * 100) / 100
        final_e_C_rounded = {key: torch.round(val * 100) / 100
                             for key, val in e_C_layers[-1].items()}

        app_h_C_final[service_name] = final_h_C_rounded.tolist()
        app_e_C_final[service_name] = {key: val.tolist()
                                       for key, val in final_e_C_rounded.items()}

        # نمایش لایه آخر
        print(f"\nFinal Layer Results for {service_name} (User: {user_id}, Helper: {helper_id}):")
        for i, h in enumerate(app_h_C_final[service_name]):
            formatted_h = [f"{x:.2f}" for x in h[:5]]
            print(f"  h_C_final[Component {i + 1}]: {formatted_h}...")
        for edge_key, e in list(app_e_C_final[service_name].items())[:10]:
            formatted_e = [f"{x:.2f}" for x in e[:5]]
            print(f"  e_C_final[{edge_key}]: {formatted_e}...")

    # ذخیره نتایج
    with open("../data/processed/msg_component1.json", "w", encoding="utf-8") as f:
        json.dump(app_h_C_final, f, indent=4)
    with open("../data/processed/msg_Cedge.json", "w", encoding="utf-8") as f:
        json.dump(app_e_C_final, f, indent=4)
    print("\nFinal embeddings saved for all services")


if __name__ == "__main__":
    msg_component_processing(sample_index=0, inner_sample_index=0)