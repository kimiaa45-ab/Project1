import yaml
import json
import os
import torch

# Load configuration from YAML file
config_path = "../configs/config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Extract configuration parameters
num_epochs = config['model']['num_epochs']
num_samples = config['model']['num_samples']
num_layers = config['model']['num_layers']
hidden_dim = config['model']['hidden_dim']
cpu_hidden_dim = config['model']['cpu_hidden_dim']
device = config['model']['device']

# Determine device and set dim accordingly
if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
        dim = hidden_dim
    else:
        device = 'cpu'
        dim = cpu_hidden_dim
else:
    device = device.lower()
    dim = hidden_dim if device == 'cuda' else cpu_hidden_dim

# Initialize lists to store data
microservices = []
microservices_edges = []
computingnodes = []
computingnodes_edges = []

# Process files in batches
i = 1
j = dim
for ins in range(1, num_samples + 1):
    if j <= num_samples:
        for x in range(i, j + 1):
            file_path = f"../data/generated/instance_{x}.json"
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Append data to respective lists
                microservices.append(data['services'])
                microservices_edges.append(data['componentConnections'])
                computingnodes.append(data['computingNodes'])
                computingnodes_edges.append(data['infraConnections'])
            else:
                print(f"Warning: File {file_path} not found.")
        i = j + 1
        j = j + dim
    else:
        # Handle remaining files if num_samples is not divisible by dim
        for x in range(i, num_samples + 1):
            file_path = f"../data/generated/instance_{x}.json"
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Append data to respective lists
                microservices.append(data['services'])
                microservices_edges.append(data['componentConnections'])
                computingnodes.append(data['computingNodes'])
                computingnodes_edges.append(data['infraConnections'])
            else:
                print(f"Warning: File {file_path} not found.")
        break

print(f"Processed {len(microservices)} instances successfully.")
print(f"Device used: {device}, Hidden dimension: {dim}")