from configs.run_config import config as run_config
from configs.model_config import config as model_config
from tool import process_data
from tool.evaluation import auc_score,list_auc_score,true_positive_rate
from models.user_model import UserModel

import datetime
import torch
import torch.nn as nn
import math
import numpy as np

from tqdm import tqdm
import time

def model_test(model,test_data,device=run_config['device']):
    test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    prediction_list = []
    model.eval()
    time.sleep(0.001)
    progress_bar = tqdm(desc="[{}] testing model".format(datetime.datetime.now()),total=len(test_data))
    for data in test_data_loader:
        [user_id, x_history, x_inview, x_global, _, label_id] = data
        out = model.sigmoid(model(x_history.to(device), x_inview.to(device), x_global.to(device)))
        prediction_list.append([user_id,out.squeeze(0).cpu().detach().numpy(),label_id.squeeze(0).detach().numpy()])
        progress_bar.update(1)
    progress_bar.close()
    return prediction_list

if __name__ == '__main__':
    #output test result file
    a=1
