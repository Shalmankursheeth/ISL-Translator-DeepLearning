import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(63, 128, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):  # x: [batch, 50, 63]
        x = x.permute(0, 2, 1)  # → [batch, 63, 50]
        x = self.encoder(x)     # → [batch, 256, 1]
        return x.squeeze(-1)    # → [batch, 256]
