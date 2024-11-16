from configs.model_config import config as model_config

from models.attention_net import AttentionModel
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
            nn.Linear(model_config['category_label_num'], output_dim*2),
            nn.BatchNorm1d(output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim*2, output_dim),
        )
        self.news_data_net = AttentionModel(output_dim+len(model_config['article_type_dict'])+len(model_config['sentiment_label_dict'])+model_config['news_vector'],output_dim*4,output_dim)
        self.feature_net = nn.Sequential(
            nn.Linear(output_dim+4, output_dim*2),
            nn.BatchNorm1d(output_dim*2),
            nn.ReLU(),
            nn.Linear(output_dim*2, output_dim),
        )
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = F.relu

    def forward(self, x_category,x_data,x_history):
        x_category = self.category_net(x_category)
        x_category = self.relu(self.bn(x_category))

        x_data = torch.cat((x_category, x_data), dim=1)
        x_data = self.news_data_net(x_data)
        x_data = self.relu(self.bn(x_data))

        x = torch.cat((x_data, x_history), dim=1)
        x = self.feature_net(x)
        return x

    def forward_on_data_feature(self,x_data,x_history):
        x_data = self.relu(self.bn(x_data))

        x = torch.cat((x_data, x_history), dim=1)
        x = self.feature_net(x)
        return x