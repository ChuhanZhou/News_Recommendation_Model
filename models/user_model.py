from configs.model_config import config as model_config

from models.user_invariant_interest_model import UserInvariantInterestModel
from models.attention_model import MLP
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class UserModel(nn.Module):
    def __init__(self,user_num=0):
        super().__init__()
        self.invariant_interest_model = UserInvariantInterestModel()

        self.bn = nn.BatchNorm1d((sum(self.invariant_interest_model.embed_setting) + model_config['pca_vector'])*2)
        self.gate = MLP(self.bn.num_features, self.bn.num_features)
        self.mlp = MLP(self.bn.num_features, self.bn.num_features)
        self.out_mlp = MLP(self.bn.num_features, 1)

        self.delta = nn.Parameter(torch.zeros(user_num+1))
        self.bce_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x_history, x_target):
        eu_H,ec = self.invariant_interest_model(x_history, x_target)

        e_concat = torch.cat((eu_H,ec),dim=2)
        e_context = self.bn(e_concat.reshape(e_concat.shape[0]*e_concat.shape[1],-1))
        e_output = self.mlp(self.gate(e_context) * e_concat.reshape(e_concat.shape[0]*e_concat.shape[1],-1))
        r = self.out_mlp(e_output).reshape(e_concat.shape[0],e_concat.shape[1])

        return r

    def loss(self,id,out,label,alpha = 0.95):
        log_loss = self.bce_loss(self.sigmoid(out),label.to(torch.float32))

        delta_imp = self.delta[id].unsqueeze(1).repeat(1, label.shape[1])
        log_loss_imp = self.bce_loss(self.sigmoid(out+delta_imp),label.to(torch.float32))

        return (1-alpha) * log_loss + alpha * log_loss_imp