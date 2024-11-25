from configs.model_config import config as model_config

from models.attention_model import PointwiseAttentionExpanded
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class UserInvariantInterestModel(nn.Module):
    def __init__(self, embed_setting=[32,16,16]):
        super().__init__()
        self.embed_setting = embed_setting
        self.slice_len_list = [
            model_config['pca_vector'],
            1,
            model_config['subcategory_max_num'],
            len(model_config['sentiment_label_dict']),
            1,
            1,
            1]
        self.category_embedding = nn.Sequential(
            nn.Embedding(model_config['category_label_num'],self.embed_setting[0]),
        )
        self.sentiment_embedding = nn.Sequential(
            nn.Linear(len(model_config['sentiment_label_dict']), self.embed_setting[1]),
            nn.ReLU()
        )
        self.type_embedding = nn.Sequential(
            nn.Embedding(len(model_config['article_type_dict']), self.embed_setting[2]),
        )
        self.w1 = nn.Linear(sum(self.embed_setting)+2,sum(self.embed_setting))

        self.label_attention = PointwiseAttentionExpanded(sum(self.embed_setting))
        self.text_img_attention = PointwiseAttentionExpanded(model_config['pca_vector'])

    def slice_x(self,x,n):
        slice_list = []
        start_i = 0
        for i in range(n):
            slice_list.append(x[:,:,start_i:start_i+self.slice_len_list[i]])
            start_i += self.slice_len_list[i]
        return slice_list

    def feature_embedding(self,category, sub_category, sentiment, type):
        category = self.category_embedding(category.reshape(category.numel(), 1).to(torch.int64)).reshape(category.shape[0], category.shape[1], -1)
        sub_category = self.category_embedding(sub_category.reshape(sub_category.numel(), 1).to(torch.int64)).reshape(sub_category.shape[0],sub_category.shape[1],sub_category.shape[2],-1).mean(dim=2)
        all_category = category + sub_category
        sentiment = self.sentiment_embedding(sentiment.reshape(sentiment.shape[0] * sentiment.shape[1], -1)).reshape(sentiment.shape[0], sentiment.shape[1], -1)
        type = self.type_embedding(type.reshape(type.numel(), 1).to(torch.int64)).reshape(type.shape[0],type.shape[1],-1)
        return torch.cat((all_category, sentiment, type), dim=2)

    def forward(self,x_history, x_target):
        [x_text_img_h, x_category_h, x_sub_category_h, x_sentiment_h, x_type_h, x_read_time_h,x_scroll_h] = self.slice_x(x_history.to(torch.float32), 7)
        [x_text_img_t, x_category_t,x_sub_category_t,x_sentiment_t,x_type_t] = self.slice_x(x_target.to(torch.float32),5)

        x_label_h = torch.cat((self.feature_embedding(x_category_h, x_sub_category_h, x_sentiment_h, x_type_h),x_read_time_h,x_scroll_h), dim=2)
        x_label_h = self.w1(x_label_h.reshape(x_label_h.shape[0]*x_label_h.shape[1],-1)).reshape(x_label_h.shape[0],x_label_h.shape[1],-1)
        x_label_t = self.feature_embedding(x_category_t, x_sub_category_t, x_sentiment_t, x_type_t)

        ec = torch.cat((x_label_t,x_text_img_t),dim=2)

        x_label_attention_score = self.label_attention(x_label_t,x_label_h)
        x_text_img_attention_score = self.text_img_attention(x_text_img_t,x_text_img_h)

        x_label = torch.sum(x_label_attention_score * x_label_h.unsqueeze(1),dim=2)
        x_text_img = torch.sum(x_text_img_attention_score * x_text_img_h.unsqueeze(1),dim=2)
        eu_H = torch.cat((x_label,x_text_img),dim=2)
        return eu_H,ec