from configs.model_config import config as model_config

from models.attention_net import AttentionModel,MultiheadAttentionModel,MultiDimensionsAttentionModel
from models.news_net import NewsModel
from models.interest_net import InterestModel
from tool import normalization

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class HistoryModel(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

        self.news_net = NewsModel(output_dim)
        self.interest_net = InterestModel(4)
        self.mlp = nn.Sequential(
            nn.Linear(model_config['news_feature']+2, model_config['news_feature']+2),
            nn.BatchNorm1d(model_config['news_feature']+2),
            nn.ReLU(),
            nn.Linear(model_config['news_feature']+2, model_config['news_feature']),
            nn.BatchNorm1d(model_config['news_feature']),
            nn.ReLU(),
        )
        self.attention = MultiDimensionsAttentionModel(model_config['news_feature'],model_config['news_feature'],16,0.2)
        self.fc_out = nn.Sequential(
            nn.Linear(model_config['news_feature'], model_config['news_feature']),
            nn.ReLU(),
            nn.Linear(model_config['news_feature'], model_config['news_feature']),
        )
        self.MSE_loss = nn.MSELoss()
        self.MR_loss = nn.MarginRankingLoss(margin=1.0)
        self.CE_loss = nn.CrossEntropyLoss()

    def forward(self,x_category,x_data,x_history,x_interest,x_time):
        history_len = model_config['history_max_length']
        x_news = self.news_net(x_category,x_data,x_history)
        x_interest = self.interest_net(x_interest)
        x_user_feature = torch.cat((x_interest,x_time,x_news), dim=1)
        x_user_feature = self.mlp(x_user_feature)
        x_user_feature = x_user_feature.view(int(x_user_feature.shape[0]/history_len), history_len, -1)
        x_user_feature = self.attention(x_user_feature)
        #x_user_feature = self.fc_out(x_user_feature)
        return x_user_feature

    def forward_on_data_feature(self,x_data,x_history,x_interest,x_time):
        history_len = model_config['history_max_length']
        x_news = self.news_net.forward_on_data_feature(x_data, x_history)
        x_interest = self.interest_net(x_interest)
        x_user_feature = torch.cat((x_interest,x_time,x_news), dim=1)
        x_user_feature = self.mlp(x_user_feature)
        x_user_feature = x_user_feature.view(int(x_user_feature.shape[0] / history_len), history_len, -1)
        x_user_feature = self.attention(x_user_feature)
        #x_user_feature = self.fc_out(x_user_feature)
        return x_user_feature

    def get_interest_rate(self,a,b):
        return (self.pearson_correlation_coefficient(a, b)+1)/2

    def pearson_correlation_coefficient(self,a,b,alpha = 1e-8):
        a_mean = torch.mean(a, dim=1, keepdim=True)
        b_mean = torch.mean(b, dim=1, keepdim=True)

        a_centered = a - a_mean
        b_centered = b - b_mean

        numerator = torch.sum(a_centered * b_centered, dim=1, keepdim=True)
        denominator = torch.sqrt(torch.sum(a_centered ** 2, dim=1, keepdim=True) * torch.sum(b_centered ** 2, dim=1, keepdim=True))

        return numerator / (denominator + alpha)

    def cos_sim(self,a,b):
        result = F.cosine_similarity(a, b, dim=1)
        return result.view(result.shape[0], 1)

    def loss(self,out,target,label,neg_list=[]):
        interest_rate = self.get_interest_rate(out, target)

        interest_loss = (1 - interest_rate).mean()
        mse_loss = self.MSE_loss(interest_rate,label)

        pos_out_list = []
        neg_out_list = []
        for neg_out in neg_list:
            neg_interest_rate = self.get_interest_rate(out, neg_out)
            pos_out_list.append(interest_rate)
            neg_out_list.append(neg_interest_rate)
        pos_out_list = torch.cat(pos_out_list,dim=0)
        neg_out_list = torch.cat(neg_out_list,dim=0)

        y = torch.ones(pos_out_list.shape).to(pos_out_list.device)

        mr_loss = self.MR_loss(pos_out_list,neg_out_list,y)

        neg_loss = neg_out_list.mean()

        loss = 1.5*mr_loss + 1.0*interest_loss + 1.0*mse_loss +2.0*neg_loss
        return loss

    def global_loss(self,out,target,neg_list=[]):
        interest_rate = self.get_interest_rate(out, target)

        neg_interest_rate = []
        for neg_out in neg_list:
            neg_interest_rate.append(self.get_interest_rate(out, neg_out))
        neg_interest_rate = torch.cat(neg_interest_rate, dim=0)
        all_interest_rate = torch.cat([interest_rate, neg_interest_rate])

        labels = torch.cat([torch.ones(interest_rate.shape,device=interest_rate.device), torch.zeros(neg_interest_rate.shape,device=neg_interest_rate.device)],dim=0)

        logloss_global = F.binary_cross_entropy(all_interest_rate, labels)

        return logloss_global