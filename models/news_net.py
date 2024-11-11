from attention_net import Attention_Net
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class News_Net(nn.Module):
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.news_net = Attention_Net(input_dim,output_dim)

    def forward(self, x):
        x = self.news_net(x)
        return x