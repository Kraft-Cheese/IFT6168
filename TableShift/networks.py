"""
Network architectures for tabular data (TableShift experiments).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularMLP(nn.Module):
    """
    Simple MLP for binary classification on tabular data.
    Matches the MLP architecture used in the TableShift benchmark.
    """

    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.1, num_classes=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
