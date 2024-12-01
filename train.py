import datetime

from configs.run_config import config as run_config
from configs.model_config import config as model_config
from tool import process_data
from tool.evaluation import auc_score
from models.user_model import UserModel
from verify import model_validation

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from tqdm import tqdm
import time

if __name__ == '__main__':
    device = run_config['device']
    epoch_num = run_config['epochs']
    batch_size = run_config['batch_size']
    lr = run_config['lr']

    training_data,max_user_id = process_data.load_processed_dataset(run_config['processed_data_path']+run_config['train_data_processed'],batch_size*1000)
    validation_data,_ = process_data.load_processed_dataset(run_config['processed_data_path']+run_config['validation_data_processed'],)

    train_data_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
    torch.manual_seed(0)

    print("[{}] device: {}".format(datetime.datetime.now(),device))
    model = UserModel(max_user_id)
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

        total_auc = 0
        total_loss = 0
        loss_str = ""
        auc_str = ""

        for i, data in enumerate(train_data_loader):
            [user_id,x_history,x_inview,x_global,label,label_id] = data
            torch.autograd.set_detect_anomaly(True)
            out = model(x_history.to(device), x_inview.to(device),x_global.to(device))

            loss = model.loss(user_id.to(device),out,label.to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            for b_i in range(user_id.shape[0]):
                auc = auc_score(label[b_i,:].detach().numpy(),out[b_i,:].cpu().detach().numpy())
                auc_str += "{:.8f}\n".format(auc)
                total_auc += auc

            total_loss += loss.item()*user_id.shape[0]
            loss_str += "{:.8f}\n".format(loss.item())
            progress_bar.set_postfix(
                loss="{:.4f}".format(loss.item()),
                loss_avg="{:.4f}".format(total_loss/((i+1)*batch_size)),
                auc_avg="{:.4f}".format(total_auc/((i+1)*batch_size)))
            progress_bar.update(1)
            #with open("./ckpt/epoch_{}.txt".format(epoch), "w", encoding="utf-8") as file:
            #    file.write(loss_str)
            #torch.save(model.state_dict(), "{}/ckpt_{}_epoch_{}.pth".format(run_config['ckpt_save_path'],run_config['train_data_processed'].split("/")[-1].split(".")[0], epoch))

        with open("./ckpt/epoch_{}.txt".format(epoch), "w", encoding="utf-8") as file:
            file.write(loss_str)
        model_ckpt = model.state_dict()
        model_ckpt.pop('delta')
        torch.save(model_ckpt, "{}/ckpt_{}_epoch_{}.pth".format(run_config['ckpt_save_path'],run_config['train_data_processed'].split("/")[-1].split(".")[0],epoch))

        #if (epoch+1)%2 == 0:
        #    scheduler.step()

        progress_bar.close()

        avg_loss = total_loss/len(training_data)
        avg_auc = total_auc/len(training_data)

        validation_result = model_validation([model],validation_data,device)
        validation_auc_score = validation_result[0]
        validation_tpr = validation_result[1]
        print("[{}] [epoch]:{} [avg_loss]:{:.3e} [auc_score_t]:{:.4f} [TPR_v]:{:.4f} [auc_score_v]:{:.4f}".format(datetime.datetime.now(),epoch,avg_loss,avg_auc,validation_tpr,validation_auc_score))

    model_ckpt = model.state_dict()
    model_ckpt.pop('delta')
    torch.save(model_ckpt, "{}/ckpt_{}_final.pth".format(run_config['ckpt_save_path'],run_config['train_data_processed'].split("/")[-1].split(".")[0]))