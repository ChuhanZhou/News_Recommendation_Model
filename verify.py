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

def model_validation(model,validation_data,device=run_config['device']):
    true_list = []
    auc = 0
    prediction_list = model_test(model, validation_data,device)
    time.sleep(0.001)
    progress_bar = tqdm(desc="[{}] validating test result".format(datetime.datetime.now()),total=len(validation_data))
    for data_i, data in enumerate(validation_data):
        [user_id, _, _, _, label, _] = data

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

if __name__ == '__main__':
    ckpt_test_list = []
    for i in range(6, 15):
        ckpt_test_list.append("./ckpt/ckpt_ebnerd_small_train_batch_epoch_{}.pth".format(i))
    # ckpt_test_list = ["./ckpt/ckpt_ebnerd_small_train_epoch_15.pth"]
    device = run_config['device']
    # device = "cpu"

    # validation_data = process_data.load_processed_dataset(run_config['validation_data_processed'])
    validation_data, _ = process_data.load_processed_dataset(run_config['processed_data_path']+"ebnerd_small_validation")

    model = UserModel()
    print("[{}] device: {}".format(datetime.datetime.now(), device))

    for ckpt_path in ckpt_test_list:
        print(ckpt_path)
        model_ckpt = torch.load(ckpt_path, map_location=torch.device(device))
        model.load_state_dict(model_ckpt, strict=False)
        model.to(device)
        auc, tpr = model_validation(model, validation_data, device)
        print("[AUC]:{:.4f} [TPR]:{:.4f}".format(auc, tpr))