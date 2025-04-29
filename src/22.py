import torch
import torch.nn as nn
import torch.nn.functional as F


class ServiceComponentGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_components: int):
        super(ServiceComponentGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_components = num_components

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Learnable matrices U_C and V_C for each layer
        self.U_C = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])
        self.V_C = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])

        # Normalization
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Component connections (6x6 directed matrix)
        self.adj_matrix = torch.tensor([
            [0, 1, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0]
        ], dtype=torch.float)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, num_services, num_components, input_dim]
        batch_size, num_services, num_components, _ = x.size()

        # Project input features
        x = F.relu(self.input_projection(x))  # [batch_size, num_services, 6, hidden_dim]

        # Message passing for each layer
        for layer in range(self.num_layers):
            h = x  # Current node features

            # Compute U_C * h_SCi
            u_term = self.U_C[layer](h)  # [batch_size, num_services, 6, hidden_dim]

            # Compute messages: ∑_{j ∈ N_{SC_i}} σ(e_SCij) ⊙ V_C * h_SCj
            v_h = self.V_C[layer](h)  # [batch_size, num_services, 6, hidden_dim]
            edge_weights = torch.sigmoid(self.adj_matrix)  # [6, 6]
            edge_weights = edge_weights.view(1, 1, 6, 6)  # [1, 1, 6, 6]
            messages = torch.matmul(edge_weights, v_h)  # [batch_size, num_services, 6, hidden_dim]

            # Combine and normalize
            updated = u_term + messages
            updated = self.norms[layer](updated)

            # Update: h_SCi^(l+1) = h_SCi^(l) + ReLU(Norm(...))
            x = h + F.relu(updated)

        return x  # [batch_size, num_services, num_components, hidden_dim]