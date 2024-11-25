from configs.run_config import config as run_config
from configs.model_config import config as model_config
from tool import process_data
from tool.evaluation import auc_score,list_auc_score,true_positive_rate
from models.user_model import UserModel

import datetime
import torch
import torch.nn as nn
import cv2
import math
import numpy as np

from tqdm import tqdm
import time

def model_validation(model,validation_data,device=run_config['device']):
    true_list = []
    auc = 0
    prediction_list = model_test(model, validation_data,device)
    time.sleep(0.001)
    progress_bar = tqdm(desc="[{}] validating test result".format(datetime.datetime.now()),total=len(validation_data))
    for data_i, data in enumerate(validation_data):
        [user_id, _, _, label, _] = data

        _, score,_ = prediction_list[data_i]

        auc += auc_score(label,score)

        if np.argmax(score) == np.argmax(label):
            true_list.append(1)
        else:
            true_list.append(0)

        progress_bar.update(1)

    progress_bar.close()
    time.sleep(0.001)
    auc = auc/len(validation_data)
    tpr = true_positive_rate(true_list)
    return [auc,tpr]

def model_test(model,test_data,device=run_config['device']):
    test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    prediction_list = []
    model.eval()
    time.sleep(0.001)
    progress_bar = tqdm(desc="[{}] testing model".format(datetime.datetime.now()),total=len(test_data))
    for data in test_data_loader:
        [user_id, x_history, x_inview, _, label_id] = data
        out = model.sigmoid(model(x_history.to(device), x_inview.to(device)))
        prediction_list.append([user_id,out.squeeze(0).cpu().detach().numpy(),label_id.squeeze(0).detach().numpy()])
        progress_bar.update(1)
    progress_bar.close()
    return prediction_list

if __name__ == '__main__':
    ckpt_test_list = []
    for i in range(20):
        ckpt_test_list.append("./ckpt/ckpt_ebnerd_small_train_epoch_{}.pth".format(i))
    #ckpt_test_list = ["./ckpt/ckpt_ebnerd_small_train_epoch_15.pth"]
    device = run_config['device']
    #device = "cpu"

    #validation_data = process_data.load_processed_dataset(run_config['validation_data_processed'])
    validation_data,_ = process_data.load_processed_dataset("./dataset/ebnerd_small_validation")

    model = UserModel()
    print("[{}] device: {}".format(datetime.datetime.now(), device))

    for ckpt_path in ckpt_test_list:
        print(ckpt_path)
        model_ckpt = torch.load(ckpt_path, map_location=torch.device(device))
        model.load_state_dict(model_ckpt, strict=False)
        model.to(device)
        auc,tpr = model_validation(model,validation_data,device)
        print("[AUC]:{:.4f} [TPR]:{:.4f}".format(auc,tpr))
