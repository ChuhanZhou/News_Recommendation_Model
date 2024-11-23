from configs.model_config import config as model_config

from models.attention_net import AttentionModel,MultiheadAttentionModel,MultiDimensionsAttentionModel
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class NewsEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.type_embedding = nn.Sequential(
            nn.Embedding(len(model_config['article_type_dict']), 64),
            nn.Linear(64, 16),
        )
        self.category_embedding = nn.Sequential(
            nn.Embedding(model_config['category_label_num'],64),
            nn.Linear(64, 16),
        )
        self.sentiment_embedding = nn.Sequential(
            nn.Linear(len(model_config['sentiment_label_dict']), 8),
        )
        self.w1 = nn.Linear(+2,)



        self.news_data_net = MultiheadAttentionModel(336,128,4,0.2)
        self.feature_net = nn.Sequential(
            nn.Linear(self.news_data_net.output_dim+4, self.news_data_net.output_dim+4),
            nn.ReLU(),
            nn.Linear(self.news_data_net.output_dim+4, output_dim),
        )
        self.bn = nn.BatchNorm1d(self.news_data_net.output_dim)
        self.relu = F.relu

    def forward(self,x_history, x_target):
        [x_text_t, x_category_t,x_sub_category_t,x_sentiment_t,x_type_t] = x_target
        [x_text_h, x_category_h, x_sub_category_h, x_sentiment_h, x_type_h,x_read_time_h,x_scroll_h] = x_history

        x_category_t = self.category_embedding(x_category_t)
        for i,x_sub_category in enumerate(x_sub_category_t):
            x_category_t[i,:] += self.category_embedding(x_sub_category).mean(dim=0, keepdim=True)
        x_type_t = self.type_embedding(x_type_t)
        x_t = torch.cat((x_category_t,x_sentiment_t,x_type_t), dim=1)

        x_category_h = self.category_embedding(x_category_h)
        for i, x_sub_category in enumerate(x_sub_category_h):
            x_category_h[i, :] += self.category_embedding(x_sub_category).mean(dim=0, keepdim=True)
        x_type_h = self.type_embedding(x_type_h)
        x_h = self.w1(torch.cat((x_category_h, x_sentiment_h, x_type_h,x_read_time_h,x_scroll_h), dim=1))


        return None

    def forward_on_data_feature(self,x_data,x_history):
        x_data = self.relu(self.bn(x_data))

        x = torch.cat((x_data, x_history), dim=1)
        x = self.feature_net(x)
        return x