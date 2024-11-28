from configs.model_config import config as model_config

from models.attention_model import PointwiseAttentionExpanded
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class UserInstantInterestModel(nn.Module):
    def __init__(self,output_dim):
        super().__init__()
        self.output_dim = output_dim

        self.out_fc = nn.Sequential(
            nn.Linear(3, output_dim),
            nn.ReLU()
        )

    def forward(self,x_global):
        eu_L = x_global.to(torch.float32)
        eu_L = self.out_fc(eu_L.reshape(eu_L.shape[0]*eu_L.shape[1],-1)).reshape(eu_L.shape[0],eu_L.shape[1],-1)
        return eu_L
