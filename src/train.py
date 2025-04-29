import yaml
import json
import os
import torch
import importlib
import shutil

# Import 1.py and 2.py dynamically
gnn1 = importlib.import_module('1')
gnn2 = importlib.import_module('2')


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    # Load configuration
    config = load_config('../configs/config.yaml')
    num_epochs = config['model']['num_epochs']
    num_samples = config['model']['num_samples']
    num_layers = config['model']['num_layers']
    device = config['model']['device']
    hidden_dim = config['model']['hidden_dim']
    cpu_hidden_dim = config['model']['cpu_hidden_dim']

    # Device selection
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_dim = hidden_dim if device.type == 'cuda' else cpu_hidden_dim
    device_str = device.type
    print(f"Running on {device_str}, hidden_dim: {hidden_dim}")

    # Calculate chunk size for for2
    chunk_size = num_samples // hidden_dim
    if num_samples % hidden_dim != 0:
        print(f"Warning: num_samples ({num_samples}) not divisible by hidden_dim ({hidden_dim})")

    # for1: Iterate over epochs
    all_results = []
    temp_sample_file = '../data/generated/sample_1.json'
    for epoch in range(1, num_epochs + 1):
        sample_file = f'../data/generated/sample_{epoch}.json'
        if not os.path.exists(sample_file):
            print(f"Error: {sample_file} not found")
            continue

        # Only copy if source and destination are different
        if sample_file != temp_sample_file:
            try:
                shutil.copyfile(sample_file, temp_sample_file)
            except Exception as e:
                print(f"Error copying {sample_file} to {temp_sample_file}: {e}")
                continue

        # Load sample data
        with open(sample_file, 'r') as f:
            samples = json.load(f)

        if not isinstance(samples, list):
            samples = [samples]

        if len(samples) != num_samples:
            print(f"Warning: Expected {num_samples} samples in {sample_file}, got {len(samples)}")

        # for2: Divide samples into chunks
        epoch_chunks = []
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i:i + chunk_size]
            if chunk:
                epoch_chunks.append(chunk)

        # for3: Process hidden_dim samples in each chunk
        epoch_results = []
        for chunk in epoch_chunks:
            for idx, sample in enumerate(chunk[:hidden_dim]):  # Limit to hidden_dim samples
                try:
                    # Process with GNN1 (components)
                    gnn1.msg_component_processing(sample_index=idx, inner_sample_index=0)
                    with open('../data/processed/msg_component1.json', 'r') as f:
                        h_C = json.load(f)
                    with open('../data/processed/msg_Cedge.json', 'r') as f:
                        e_C = json.load(f)

                    # Process with GNN2 (nodes)
                    gnn2.msg_node_processing(sample_index=idx, inner_sample_index=0)
                    with open('../data/processed/msg_node1.json', 'r') as f:
                        h_N = json.load(f)
                    with open('../data/processed/msg_Nedge1.json', 'r') as f:
                        e_N = json.load(f)

                    # Store results
                    epoch_results.append({
                        'epoch': epoch,
                        'sample': sample,
                        'component_node_embeddings': h_C,
                        'component_edge_embeddings': e_C,
                        'node_embeddings': h_N,
                        'node_edge_embeddings': e_N
                    })
                except Exception as e:
                    print(f"Error processing sample in epoch {epoch}: {e}")
                    continue

        all_results.append(epoch_results)
        print(f"Epoch {epoch} processed, {len(epoch_results)} samples done")

        # Clean up temporary file if it was copied
        if sample_file != temp_sample_file and os.path.exists(temp_sample_file):
            try:
                os.remove(temp_sample_file)
            except Exception as e:
                print(f"Error removing {temp_sample_file}: {e}")

    print(f"Total results: {len(all_results)} epochs processed")
    return all_results


if __name__ == "__main__":
    main()