from configs.model_config import config as model_config

from models.attention_net import AttentionModel,MultiheadAttentionModel,MultiDimensionsAttentionModel
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class NewsModel(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

        self.category_net = nn.Sequential(
            nn.Linear(model_config['category_label_num'], model_config['category_label_num']),
            nn.ReLU(),
            nn.Linear(model_config['category_label_num'], 45),
            nn.ReLU(),
        )
        self.news_data_net = MultiheadAttentionModel(128,64,4,0.2)
        self.feature_net = nn.Sequential(
            nn.Linear(self.news_data_net.output_dim+4, self.news_data_net.output_dim+4),
            nn.ReLU(),
            nn.Linear(self.news_data_net.output_dim+4, output_dim),
        )
        self.bn = nn.BatchNorm1d(self.news_data_net.output_dim)
        self.relu = F.relu

    def forward(self, x_category,x_data,x_history):
        x_category = self.category_net(x_category)

        x_data = torch.cat((x_category, x_data), dim=1)
        x_data = self.news_data_net(x_data)
        x_data = self.relu(x_data)

        x = torch.cat((x_data, x_history), dim=1)
        x = self.feature_net(x)
        return x

    def forward_on_data_feature(self,x_data,x_history):
        x_data = self.relu(self.bn(x_data))

        x = torch.cat((x_data, x_history), dim=1)
        x = self.feature_net(x)
        return x