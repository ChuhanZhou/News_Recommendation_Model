import datetime

from configs.run_config import config as run_config
from configs.model_config import config as model_config
from tool import process_data
from models.history_net import HistoryModel
from test import model_validation

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import math
import numpy as np

from tqdm import tqdm
import time

if __name__ == '__main__':
    device = run_config['device']
    epoch_num = run_config['epochs']
    batch_size = run_config['batch_size']
    lr = run_config['lr']

    #training_data = process_data.load_processed_dataset(run_config['train_data_processed'],1024000)
    training_data = process_data.load_processed_dataset(run_config['train_data_processed'],batch_size*1000)
    validation_data_0 = process_data.load_processed_dataset("./dataset/ebnerd_demo_t.validation", 2500)
    validation_data = process_data.load_processed_dataset(run_config['validation_data_processed'],2500)

    training_processed_data,training_category_data,training_news_info_data = training_data

    train_data_loader = torch.utils.data.DataLoader(dataset=training_processed_data, batch_size=batch_size, shuffle=True)
    #torch.manual_seed(0)

    print("[{}] device: {}".format(datetime.datetime.now(),device))
    model = HistoryModel(model_config['news_feature'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.65 ** epoch)

    model.train()
    print("training start")
    print("epoch num: {} | batch size: {} | init lr: {:.3e}".format(epoch_num, batch_size, lr))

    for epoch in range(epoch_num):
        model.train()
        lr = optimizer.param_groups[0]['lr']
        time.sleep(0.001)
        progress_bar = tqdm(desc="[{}] [epoch]:{} [lr]:{:.3e}".format(datetime.datetime.now(),epoch,lr),total=len(train_data_loader))
        total_loss = 0
        loss_str = ""

        for i, data in enumerate(train_data_loader):
            user_history,target,label_feature,neg_dim_lists = process_data.build_full_running_data(data,training_category_data,training_news_info_data)
            x_category, x_data, x_history, x_interest, x_time = user_history
            out_x = model(x_category.to(device), x_data.to(device), x_history.to(device), x_interest.to(device), x_time.to(device))

            t_category,t_data,t_history = target
            out_t = model.news_net(t_category.to(device),t_data.to(device),t_history.to(device))

            out_i = model.interest_net(label_feature.to(device))

            out_n_list = []
            for n_dim in neg_dim_lists:
                n_category, n_data, n_history = n_dim
                out_n = model.news_net(n_category.to(device), n_data.to(device), n_history.to(device))
                out_n_list.append(out_n)

            #loss = model.loss(out_x,out_t,out_i,out_n_list)
            loss = model.global_loss(out_x,out_t,out_n_list) + model.loss(out_x,out_t,out_i,out_n_list)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            loss_str += "{:.8f}\n".format(loss.item())
            progress_bar.update(1)
            with open("./ckpt/epoch_{}.txt".format(epoch), "w", encoding="utf-8") as file:
                file.write(loss_str)
            torch.save(model.state_dict(), "{}/ckpt_{}_epoch_{}.pth".format(run_config['ckpt_save_path'],run_config['train_data_processed'].split("/")[-1].split(".")[0], epoch))

        progress_bar.close()
        avg_loss = total_loss/len(train_data_loader)
        #scheduler.step()

        validation_result = model_validation(model,validation_data,device)
        validation_auc_score = validation_result[0]
        validation_tpr = validation_result[1]

        train_result = model_validation(model, validation_data_0, device)
        train_auc_score = train_result[0]
        train_tpr = train_result[1]

        with open("./ckpt/epoch_{}.txt".format(epoch), "w", encoding="utf-8") as file:
            file.write(loss_str)
        torch.save(model.state_dict(), "{}/ckpt_{}_epoch_{}.pth".format(run_config['ckpt_save_path'],run_config['train_data_processed'].split("/")[-1].split(".")[0],epoch))
        print("[{}] [epoch]:{} [avg_loss]:{:.3e} [TPR_t]:{:.4f} [auc_score_t]:{:.4f} [TPR_v]:{:.4f} [auc_score_v]:{:.4f}".format(datetime.datetime.now(),epoch,avg_loss,train_tpr,train_auc_score,validation_tpr,validation_auc_score))

    torch.save(model.state_dict(), "{}/ckpt_{}_final.pth".format(run_config['ckpt_save_path'],run_config['train_data_processed'].split("/")[-1].split(".")[0]))