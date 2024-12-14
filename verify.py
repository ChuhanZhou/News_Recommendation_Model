from configs.run_config import config as run_config
from configs.model_config import config as model_config
from tool import process_data
from tool.evaluation import auc_score,list_auc_score,true_positive_rate
from models.user_model import UserModel
from test import model_test

import datetime
import torch
import torch.nn as nn
import math
import numpy as np

from tqdm import tqdm
import time

import argparse

def model_validation(model_list,validation_data,device=run_config['device'],batch_size=1):
    true_list = []
    auc = 0
    prediction_queue,id_list = model_test(model_list, validation_data,device,batch_size=batch_size)
    time.sleep(0.001)
    progress_bar = tqdm(desc="[{}] validating test result".format(datetime.datetime.now()),total=len(validation_data))
    for data_i, data in enumerate(validation_data):
        [impression_id,user_id, _, _, _, label, _, _] = data

        _,_,score,_ = prediction_queue.get()

        auc += auc_score(label[0:len(score)],score)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify model")
    parser.add_argument('--data', help="validation dataset path",default=run_config['processed_data_path'] + run_config['validation_data_processed'])
    parser.add_argument('--batch', help="batch size", type=int, default=80)
    parser.add_argument('--model', help="model parameter file", default=run_config['ckpt_save_path'] + "ckpt_ebnerd_large_train_batch_epoch_{}.pth")
    parser.add_argument('--ckpt', help="check point number", type=int, default=5)
    args = parser.parse_args()

    ckpt_test_list = []
    for i in range(0, args.ckpt):
        ckpt_test_list.append([args.model.format(i)])

    validation_data, _ = process_data.load_processed_dataset(args.data)

    device = run_config['device']
    print("[{}] device: {}".format(datetime.datetime.now(), device))

    best_parameter_info = ["",0]
    for ckpt_path_list in ckpt_test_list:
        model_list = []
        for ckpt_path in ckpt_path_list:
            print(ckpt_path)
            model = UserModel()
            model.load_state_dict(torch.load(ckpt_path), strict=False)
            model.to(device)
            model_list.append(model)
        auc, tpr = model_validation(model_list, validation_data, device,args.batch)
        print("[AUC]:{:.8f} [TPR]:{:.8f}".format(auc, tpr))
        if auc>=best_parameter_info[1]:
            best_parameter_info = [ckpt_path_list,auc]
    print("[{}] best auc score: {:.8f} | best parameter path: {}".format(datetime.datetime.now(),best_parameter_info[1],best_parameter_info[0]))