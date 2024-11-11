


import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np

class Attention(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim):
        super().__init__()
        self.query_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.key_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.out_layer = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        query = self.query_mlp(x)
        key = self.key_mlp(x)
        value = self.value_mlp(x)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(key.size(-1), dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)

        weighted_sum = torch.matmul(attention_weights, value)

        out = self.out_layer(weighted_sum)
        return out

class Attention_Net(nn.Module):
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.attention_layer0 = Attention(input_dim,output_dim*8,output_dim*4)
        self.attention_layer1 = Attention(output_dim*4,output_dim*2,output_dim)

        self.bn = nn.BatchNorm1d(output_dim*4)

    def forward(self, x):
        x = self.attention_layer0(x)
        x = self.bn(x)
        x = self.attention_layer1(x)
        return x