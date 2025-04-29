import json
import os
import random
import uuid

# Ensure the output directory exists
output_dir = "../data/generated"
os.makedirs(output_dir, exist_ok=True)

# Fixed component connections from the input file
component_connections = [
    [0, 1, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0]
]

# Results section copied from the input file
results = {
    "algorithm": "TCA",
    "totalResponseTime": 0.8927,
    "aveResponseTime": 0.1785,
    "platformReliability": 1,
    "serviceReliability": 1,
    "infrastructureMemoryEntropy": 2.5628,
    "infrastructureCPUEntropy": 2.3998,
    "algorithmRuntime": 0.3228,
    "responseTimePerService": {
        "resTimePerService": {
            "service1": 0.192,
            "service2": 0.124,
            "service3": 0.1345,
            "service4": 0.28,
            "service5": 0.1622
        },
        "platReliability": {
            "service1": 1,
            "service2": 1,
            "service3": 1,
            "service4": 1,
            "service5": 1
        },
        "servReliability": {
            "service1": 1,
            "service2": 1,
            "service3": 1,
            "service4": 1,
            "service5": 1
        }
    },
    "finalSolution": [
        {"serviceID": 1, "componentID": 1, "versionID": 1, "nodeID": 10},
        {"serviceID": 1, "componentID": 2, "versionID": 1, "nodeID": 1},
        {"serviceID": 1, "componentID": 3, "versionID": 1, "nodeID": 10},
        {"serviceID": 1, "componentID": 4, "versionID": 1, "nodeID": 1},
        {"serviceID": 1, "componentID": 5, "versionID": 1, "nodeID": 1},
        {"serviceID": 1, "componentID": 6, "versionID": 1, "nodeID": 7},
        {"serviceID": 2, "componentID": 1, "versionID": 1, "nodeID": 11},
        {"serviceID": 2, "componentID": 2, "versionID": 1, "nodeID": 2},
        {"serviceID": 2, "componentID": 3, "versionID": 1, "nodeID": 2},
        {"serviceID": 2, "componentID": 4, "versionID": 1, "nodeID": 3},
        {"serviceID": 2, "componentID": 5, "versionID": 1, "nodeID": 3},
        {"serviceID": 2, "componentID": 6, "versionID": 1, "nodeID": 8},
        {"serviceID": 3, "componentID": 1, "versionID": 1, "nodeID": 12},
        {"serviceID": 3, "componentID": 2, "versionID": 1, "nodeID": 4},
        {"serviceID": 3, "componentID": 3, "versionID": 1, "nodeID": 4},
        {"serviceID": 3, "componentID": 4, "versionID": 1, "nodeID": 3},
        {"serviceID": 3, "componentID": 5, "versionID": 1, "nodeID": 4},
        {"serviceID": 3, "componentID": 6, "versionID": 1, "nodeID": 9},
        {"serviceID": 4, "componentID": 1, "versionID": 1, "nodeID": 13},
        {"serviceID": 4, "componentID": 2, "versionID": 1, "nodeID": 4},
        {"serviceID": 4, "componentID": 3, "versionID": 1, "nodeID": 13},
        {"serviceID": 4, "componentID": 4, "versionID": 1, "nodeID": 4},
        {"serviceID": 4, "componentID": 5, "versionID": 1, "nodeID": 4},
        {"serviceID": 4, "componentID": 6, "versionID": 1, "nodeID": 7},
        {"serviceID": 5, "componentID": 1, "versionID": 1, "nodeID": 14},
        {"serviceID": 5, "componentID": 2, "versionID": 1, "nodeID": 5},
        {"serviceID": 5, "componentID": 3, "versionID": 1, "nodeID": 5},
        {"serviceID": 5, "componentID": 4, "versionID": 1, "nodeID": 5},
        {"serviceID": 5, "componentID": 5, "versionID": 1, "nodeID": 5},
        {"serviceID": 5, "componentID": 6, "versionID": 1, "nodeID": 8}
    ]
}


# Generate one instance
def generate_instance():
    instance = {
        "comment": "CPU unit = MIPS, Mem unit = GB, Disk unit = GB, BW = Mbps, Datasize = Mb, Response time = second",
        "computingNodes": [],
        "helperNodes": [],
        "usersNodes": [],
        "services": [],
        "componentConnections": component_connections,
        "infraConnections": [],
        "results": results
    }

    # Generate computingNodes (20 nodes)
    for i in range(20):
        node_id = i + 1
        if i < 10:  # Tier 1
            node_tier = 1
            cpu = random.randint(1500, 2200)
            memory = random.choice([4, 8])
            disk = random.choice([8, 16, 32])
        elif i < 18:  # Tier 2
            node_tier = 2
            cpu = random.randint(5000, 15000)
            memory = random.choice([8, 16])
            disk = random.choice([32, 64, 128])
        else:  # Tier 3
            node_tier = 3
            cpu = random.randint(15000, 30000)
            memory = random.choice([32, 64])
            disk = random.choice([128, 256])

        instance["computingNodes"].append({
            "nodeID": node_id,
            "nodeTier": node_tier,
            "characteristics": {
                "cpu": cpu,
                "memory": memory,
                "disk": disk,
                "reliabilityScore": 1
            }
        })

    # Generate helperNodes (8 nodes)
    for i in range(8):
        node_id = 21 + i
        instance["helperNodes"].append({
            "nodeID": node_id,
            "nodeTier": 4,
            "characteristics": {
                "cpu": random.randint(1500, 2500),
                "memory": random.choice([2, 4]),
                "disk": random.choice([4, 8]),
                "reliability": 1
            }
        })

    # Generate usersNodes (15 nodes)
    for i in range(15):
        node_id = 29 + i
        instance["usersNodes"].append({
            "nodeID": node_id,
            "nodeTier": 0,
            "characteristics": {
                "cpu": random.randint(500, 2200),
                "memory": random.choice([2, 4]),
                "disk": random.choice([4, 8]),
                "reliability": 1
            }
        })

    # Generate services (15 services, each with 5 components)
    for i in range(15):
        service_id = i + 1
        components = []
        for j in range(5):
            component_id = j + 1
            components.append({
                "componentID": component_id,
                    "characteristics": {
                        "cpu": random.randint(800, 3000),
                        "memory": round(random.uniform(1.5, 3.3), 1),
                        "dataSize": random.randint(500, 800),
                        "disk": round(random.uniform(1, 3), 1),
                        "reliabilityScore": 1
                    }
            })
        instance["services"].append({
            "serviceID": service_id,
            "components": components,
            "userID": random.randint(29, 43),  # Random user node ID
            "helperID": random.randint(21, 28)  # Random helper node ID
        })

    # Generate infraConnections (44x44 matrix)
    total_nodes = 44
    infra_connections = []
    for i in range(total_nodes):
        row = []
        for j in range(total_nodes):
            # No connections between tier 4 nodes, tier 0 nodes, or between tier 0 and tier 4
            if (i == j or
                    (instance["computingNodes"][i]["nodeTier"] == 4 and instance["computingNodes"][j][
                        "nodeTier"] == 4 if i < 20 and j < 20 else False) or
                    (instance["usersNodes"][i - 29]["nodeTier"] == 0 and instance["usersNodes"][j - 29][
                        "nodeTier"] == 0 if i >= 29 and j >= 29 else False) or
                    ((instance["usersNodes"][i - 29]["nodeTier"] == 0 if i >= 29 else False) and (
                    instance["helperNodes"][j - 21]["nodeTier"] == 4 if j >= 21 and j < 29 else False)) or
                    ((instance["helperNodes"][i - 21]["nodeTier"] == 4 if i >= 21 and i < 29 else False) and (
                    instance["usersNodes"][j - 29]["nodeTier"] == 0 if j >= 29 else False))):
                row.append([0, 0])
            else:
                bandwidth = round(random.uniform(100, 500), 2)
                delay = round(random.uniform(0, 1), 2)
                row.append([bandwidth, delay])
        infra_connections.append(row)
    instance["infraConnections"] = infra_connections

    return instance


# Generate 1024 instances
for i in range(1, 1025):
    instance = generate_instance()
    output_file = os.path.join(output_dir, f"instance_{i}.json")
    with open(output_file, 'w') as f:
        json.dump(instance, f, indent=2)
    print(f"Generated {output_file}")

print("All 1024 instances generated successfully.")