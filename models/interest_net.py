import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class Interest_Net(nn.Module):
    def __init__(self,input_dim=2, hidden_dim=8):
        super().__init__()
        self.layer0 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.relu = F.relu

    def forward(self, x):
        x = self.layer0(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.out_layer(x)
        return x