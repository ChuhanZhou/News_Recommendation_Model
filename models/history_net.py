from attention_net import Attention_Net
from news_net import News_Net
from interest_net import Interest_Net
from tool import normalization

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class Net(nn.Module):
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.news_net = News_Net(input_dim, output_dim)
        self.interest_net = Interest_Net()
        self.attention = Attention_Net()

    def forward(self, x):
        interest_x, impression_time, category_x,article_data,article_history,now
        news_x = self.news_net(category_x,article_data,article_history)
        interest_x = self.interest_net(interest_net)
        impression_time_norm = normalization.datetime_norm(impression_time,now)
        x = self.attention(interest_x,impression_time_norm，news_x)
        return x

    def loss(self, ):

        return