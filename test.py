from configs.run_config import config as run_config
from configs.model_config import config as model_config
from tool import process_data
from models.history_net import HistoryModel
from tool.evaluation import auc_score,true_positive_rate

import datetime
import torch
import torch.nn.functional as F
import cv2
import math
import numpy as np

from tqdm import tqdm
import time

def model_validation(model,validation_data,device=run_config['device']):
    validation_processed_data, _, _ = validation_data

    true_list = []
    score_list = []
    prediction_list = model_test(model, validation_data,device)
    time.sleep(0.001)
    progress_bar = tqdm(desc="[{}] validating test result".format(datetime.datetime.now()),total=len(validation_processed_data))
    for data_i, processed_data in enumerate(validation_processed_data):
        _, _, label_id = processed_data

        sorted_id, sorted_int = prediction_list[data_i]

        if sorted_id[0] == label_id:
            true_list.append(1)
        else:
            true_list.append(0)
        score_list.append((sorted_int[0] + 1) / 2)

        #for id_i, id in enumerate(sorted_id):
        #    sim = sorted_sim[id_i]
        #    if id == label_id:
        #        true_list.append(1)
        #    else:
        #        true_list.append(0)
        #    score_list.append((sim+1)/2)

        progress_bar.update(1)
    progress_bar.close()
    time.sleep(0.001)
    auc = auc_score(true_list,score_list)
    tpr = true_positive_rate(true_list)
    return [auc,tpr]

def model_test(model,test_data,device=run_config['device']):
    test_processed_data, test_category_data, test_news_info_data = test_data
    model.eval()

    news_data_dict = {}
    time.sleep(0.001)
    progress_bar = tqdm(desc="[{}] reducing news data dimensions".format(datetime.datetime.now()),total=len(test_category_data))
    for news_id in test_category_data:
        x_category = torch.tensor(test_category_data[news_id]).view(1, model_config['category_label_num']).float().to(device)
        x_data = torch.tensor(test_news_info_data[news_id]).view(1, len(model_config['article_type_dict']) + len(model_config['sentiment_label_dict']) + model_config['news_vector']).float().to(device)

        x_category = model.news_net.category_net(x_category)
        x_data = torch.cat((x_category, x_data), dim=1)
        x_data = model.news_net.news_data_net(x_data)
        news_data_dict[news_id] = x_data.view(model_config['news_feature']).to("cpu")
        progress_bar.update(1)
    progress_bar.close()

    prediction_list = []
    time.sleep(0.001)
    progress_bar = tqdm(desc="[{}] testing model".format(datetime.datetime.now()),total=len(test_processed_data))
    for processed_data in test_processed_data:
        test_history_feature_list, inview_news_feature_list= processed_data[0:2]
        history_feature, inview_feature, inview_id_list = build_test_running_data(test_history_feature_list,inview_news_feature_list,news_data_dict)

        x_data, x_history, x_interest, x_time = history_feature
        out_x = model.forward_on_data_feature(x_data.to(device), x_history.to(device), x_interest.to(device),x_time.to(device))

        t_data, t_history = inview_feature
        out_t = model.news_net.forward_on_data_feature(t_data.to(device), t_history.to(device))

        interest_rate = model.get_interest_rate(out_x.repeat(out_t.shape[0], 1), out_t).squeeze().tolist()
        sorted_id, sorted_int = zip(*sorted(zip(inview_id_list, interest_rate), key=lambda x: x[1], reverse=True))
        prediction_list.append([sorted_id, sorted_int])
        progress_bar.update(1)
    progress_bar.close()
    return prediction_list

def build_test_running_data(history_feature_list, inview_news_history_list,news_data_dict):
    history_interest = []
    history_timestamp = []
    history_news_data = []
    history_news_history = []

    inview_id_list = []
    inview_news_data = []
    inview_news_history = []

    for history_feature in history_feature_list:
        news_id = int(history_feature[0])
        interest = history_feature[1:3]
        timestamp = history_feature[3]
        news_history = history_feature[4:9]

        news_data = torch.zeros((model_config['news_feature']))
        if news_id in news_data_dict:
            news_data = news_data_dict[news_id]

        history_interest.append(torch.Tensor(interest))
        history_timestamp.append(torch.Tensor([timestamp]))
        history_news_data.append(news_data)
        history_news_history.append(torch.Tensor(news_history))

    for history_feature in inview_news_history_list:
        news_id = int(history_feature[0])
        news_history = history_feature[1:5]

        news_data = news_data_dict[news_id]

        inview_id_list.append(news_id)
        inview_news_data.append(news_data)
        inview_news_history.append(torch.Tensor(news_history))

    history_interest = torch.stack(history_interest).float()
    history_timestamp = torch.stack(history_timestamp).float()
    history_news_data = torch.stack(history_news_data).float()
    history_news_history = torch.stack(history_news_history).float()
    inview_news_data = torch.stack(inview_news_data).float()
    inview_news_history = torch.stack(inview_news_history).float()

    history_feature = [history_news_data,history_news_history,history_interest,history_timestamp]
    inview_feature = [inview_news_data,inview_news_history]

    return history_feature,inview_feature,inview_id_list

if __name__ == '__main__':
    ckpt_test_list = [
        #"./ckpt/ckpt_ebnerd_demo_epoch_0.pth",
        #"./ckpt/ckpt_ebnerd_demo_epoch_1.pth",
        #"./ckpt/ckpt_ebnerd_demo_epoch_2.pth",
        #"./ckpt/ckpt_ebnerd_demo_epoch_3.pth",
        #"./ckpt/ckpt_ebnerd_demo_epoch_4.pth",
        #"./ckpt/ckpt_ebnerd_demo_epoch_5.pth",
    ]
    ckpt_test_list = ["./ckpt/ckpt_ebnerd_demo_epoch_0.pth"]
    device = run_config['device']
    #device = "cpu"

    validation_data = process_data.load_processed_dataset(run_config['validation_data_processed'],)

    model = HistoryModel(model_config['news_feature'])
    print("[{}] device: {}".format(datetime.datetime.now(), device))

    for ckpt_path in ckpt_test_list:
        print(ckpt_path)

        model_ckpt = torch.load(ckpt_path, map_location=torch.device(device))
        model.load_state_dict(model_ckpt, strict=True)
        model.to(device)
        print(model_validation(model,validation_data,device))
