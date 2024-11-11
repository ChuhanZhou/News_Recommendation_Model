from attention_net import Attention_Net
from news_net import News_Net
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
        self.attention_net = Attention_Net(input_dim,output_dim)

    def forward(self, x):
        x = self.attention_net(x)
        return x

    def loss(self, ):

        return