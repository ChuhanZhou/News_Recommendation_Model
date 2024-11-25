import torch
import torch.nn as nn
import numpy as np
import pickle

# e_i = target article
# e_j = historical data, dimensionally aligned with e_i
# Attention = MLP(𝒆 𝑗, 𝒆𝑖, 𝒆𝑖 − 𝒆 𝑗, 𝒆𝑖 ⊙ 𝒆 𝑗) = MLP(concat(historical, target, target - historical, target * historical (pointwise)))

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, activation_type='gelu'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 4)
        self.fc2 = nn.Linear(input_dim // 4, output_dim)
        self.activation_type = activation_type
        self.activation = self._get_activation()

    def _get_activation(self):
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        return activations.get(self.activation_type.lower(), nn.GELU())

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        return self.fc2(x)

class PointwiseAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = MLP(input_dim * 4, 1)

    def forward(self, target, history):
        # Create interaction features as described in paper
        diff = target - history
        pw_prod = target * history
        concat = torch.cat([history, target, diff, pw_prod], dim=-1)
        return self.mlp(concat)


class PointwiseAttentionExpanded(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = MLP(input_dim * 4, 1)

    def forward(self, target, history):
        """
        Compute pointwise attention scores for batched inputs

        Args:
            target: Shape [batch_size, 1, embed_dim] or [batch_size, num_targets, embed_dim]
            history: Shape [batch_size, num_history, embed_dim]

        Returns:
            attention_scores: Shape [batch_size, num_targets, num_history, 1]
        """
        # Add target sequence dimension if needed
        if len(target.shape) == 2:
            target = target.unsqueeze(1)  # [batch_size, 1, embed_dim]

        batch_size, num_targets, embed_dim = target.shape
        num_history = history.shape[1]

        # Reshape target to align with history for broadcasting,  [batch_size, num_targets, 1, embed_dim]
        target_expanded = target.unsqueeze(2)

        # Reshape history to align with target for broadcasting, [batch_size, 1, num_history, embed_dim]
        history_expanded = history.unsqueeze(1)

        # Create interaction features, each will be [batch_size, num_targets, num_history, embed_dim]
        diff = target_expanded - history_expanded
        pw_prod = target_expanded * history_expanded

        # Concatenate along feature dimension, [batch_size, num_targets, num_history, embed_dim * 4]
        concat = torch.cat([
            history_expanded.expand(-1, num_targets, -1, -1),
            target_expanded.expand(-1, -1, num_history, -1),
            diff,
            pw_prod
        ], dim=-1)

        # Reshape for MLP, [batch_size * num_targets * num_history, embed_dim * 4]
        flat_concat = concat.view(-1, embed_dim * 4)

        # Apply MLP, [batch_size * num_targets * num_history, 1]
        scores = self.mlp(flat_concat)

        # Reshape back, [batch_size, num_targets, num_history, 1]
        attention_scores = scores.view(batch_size, num_targets, num_history, 1)

        return attention_scores