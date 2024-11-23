import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class InterestModel(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.fc_0 = nn.Linear(2, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.relu = F.relu
        self.sigmoid = F.sigmoid

    def forward(self, x):
        x = self.fc_0(x)
        x = self.relu(x)
        out = self.fc_out(x)
        out = self.sigmoid(out)
        return out