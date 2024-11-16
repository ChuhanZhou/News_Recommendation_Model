
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np

class AttentionModel(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.query_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
        )
        self.key_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = F.relu
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        query = self.query_mlp(x)
        key = self.key_mlp(x)
        value = self.value_mlp(x)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.fc_out.in_features ** 0.5)
        weights = F.softmax(scores, dim=-1)

        x = torch.matmul(weights, value)
        #x = self.relu(self.bn(x))
        out = self.fc_out(x)
        return out

class MultiFeatureAttentionModel(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.query_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.key_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = F.relu
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        query = self.query_mlp(x.view(-1, x.shape[-1]))
        query = query.view(x.shape[0],x.shape[1],-1)
        key = self.key_mlp(x.view(-1, x.shape[-1]))
        key = key.view(x.shape[0], x.shape[1], -1)
        value = self.value_mlp(x.view(-1, x.shape[-1]))
        value = value.view(x.shape[0], x.shape[1], -1)

        scores = torch.matmul(query, key.transpose(-2, -1))
        weights = F.softmax(scores, dim=-1)

        x = torch.matmul(weights, value)
        x = x.sum(dim=1, keepdim=False)
        # x = self.relu(self.bn(x))
        out = self.fc_out(x)
        return out

class MultiheadAttentionModel(nn.Module):
    def __init__(self, input_dim, output_dim,heads_num=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=heads_num)
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(input_dim)
        self.relu = F.relu

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x_attention, _ = self.attention(x, x, x)
        x_attention = x_attention.mean(dim=0, keepdim=True)
        x_attention = x_attention.view(x_attention.shape[1],x_attention.shape[2])
        #x_attention = self.relu(self.bn(x_attention))
        output = self.fc_out(x_attention)
        return output
