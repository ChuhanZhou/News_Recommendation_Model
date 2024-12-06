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
import zipfile
from tqdm import tqdm
import time
from torch.multiprocessing import Manager,Process,Pool

import argparse

def model_test_in_thread(test_data_path_list,model_list,batch_size,device,prediction_list,status):
    test_data = []
    for path in test_data_path_list:
        part_test_data = process_data.import_processed_data(path)
        print("load {} data from {}".format(len(part_test_data), path))
        test_data+=part_test_data
    model_test(model_list,test_data,device,prediction_list,batch_size)
    status.put(0)

@torch.no_grad()
def model_test(model_list,test_data,device=run_config['device'],prediction_list=None,batch_size=1):
    test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    for model in model_list:
        model.eval()
        model.to(device)

    if prediction_list is None:
        prediction_list = []
    time.sleep(0.001)
    progress_bar = tqdm(desc="[{}] testing model".format(datetime.datetime.now()), total=len(test_data))
    softmax = nn.Softmax(dim=1)
    for data in test_data_loader:
        [impression_id,user_id, x_history, x_inview, x_global, _, label_id, empty_num] = data
        min_zero_data_num = torch.min(empty_num)
        if min_zero_data_num>0:
            x_inview = x_inview[:,0:-min_zero_data_num]
            x_global = x_global[:,0:-min_zero_data_num]
            label_id = label_id[:,0:-min_zero_data_num]
            empty_num = empty_num-min_zero_data_num

        out = None
        for model in model_list:
            if out is None:
                out = softmax(model(x_history.to(device), x_inview.to(device), x_global.to(device))).cpu()
            else:
                out += softmax(model(x_history.to(device), x_inview.to(device), x_global.to(device))).cpu()
        out = out/len(model_list)
        for d_i in range(out.shape[0]):
            zero_data_num = int(empty_num[d_i])
            if zero_data_num>0:
                prediction_list.append([impression_id[d_i],softmax(out[d_i:d_i+1,0:-zero_data_num]).squeeze(0).detach().numpy(),label_id[d_i,0:-zero_data_num].detach().numpy()])
            else:
                prediction_list.append([impression_id[d_i], out[d_i].detach().numpy(),label_id[d_i].detach().numpy()])
            progress_bar.update()
    progress_bar.close()
    return prediction_list

def write_submission_file(prediction_list,path=run_config['output_path'],name="predictions",thread_num=run_config['thread_num']):
    file_path = path + "predictions.txt"
    zip_path = path + "{}.zip".format(name)

    process_task_list = []
    return_queue_list = []

    start_data_i = 0
    time.sleep(0.001)
    progress_bar = tqdm(total=len(prediction_list), desc="[{}] predict result to string".format(datetime.datetime.now()))
    bar_queue = Manager().Queue(len(prediction_list))
    for i in range(thread_num):
        if i + 1 != thread_num:
            data_num = math.ceil(len(prediction_list) / thread_num)
        else:
            data_num = len(prediction_list) - start_data_i
        return_queue = Manager().Queue(1)
        process_task = Process(target=get_string_of_prediction,args=[prediction_list,start_data_i,data_num,return_queue, bar_queue])
        process_task.start()
        process_task_list.append(process_task)
        return_queue_list.append(return_queue)
        start_data_i += data_num

    total_process = 0
    while (total_process != len(prediction_list)):
        time.sleep(0.2)
        now_process = bar_queue.qsize()
        progress_bar.update(now_process - total_process)
        total_process = now_process
    progress_bar.close()

    for process_task in process_task_list:
        process_task.terminate()

    submit_info = ""
    time.sleep(0.001)
    progress_bar = tqdm(desc="[{}] mix process result".format(datetime.datetime.now()), total=len(return_queue_list))
    for return_queue in return_queue_list:
        part_submit_info = return_queue.get()
        submit_info += part_submit_info
        progress_bar.update()
    progress_bar.close()

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(submit_info)

    zip_file = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)
    zip_file.write(file_path,arcname=file_path.split("/")[-1])
    zip_file.close()

def get_string_of_prediction(prediction_list,start,num,return_queue,bar_queue):
    submit_info = ""
    prediction_list = prediction_list[start:start+num]
    for impression_id,prediction_scores,ids in prediction_list:
        score_info = ""
        for score in prediction_scores:
            score_info += "{},".format(score)
        score_info = score_info[0:len(score_info)-1]
        submit_info += "{} [{}]\n".format(int(impression_id),score_info)
        bar_queue.put(0)
    return_queue.put(submit_info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test model")
    parser.add_argument('--data', help="test dataset path", default=run_config['processed_data_path'] + run_config['test_data_processed'])
    parser.add_argument('--volume', help="how many data volumes to read together", type=int, default=1)
    parser.add_argument('--batch', help="batch size", type=int, default=80)
    parser.add_argument('--out', help="result output path", default=run_config['output_path'])
    parser.add_argument('--model_0', help="model parameter file 0", default=run_config['ckpt_save_path'] + "ckpt_ebnerd_large_train_final.pth")
    parser.add_argument('--model_1', help="model parameter file 1", default=run_config['ckpt_save_path'] + "ckpt_ebnerd_large_validation_final.pth")
    parser.add_argument('--thread', help="how may thread can be used", type=int, default=run_config['thread_num'])
    args = parser.parse_args()

    run_config['thread_num'] = args.thread
    head_file_path = args.data
    part_data_num = args.volume
    batch_size = args.batch
    device = run_config['device']
    print("[{}] device: {}".format(datetime.datetime.now(), device))
    model_t = UserModel()
    model_t.load_state_dict(torch.load(args.model_0), strict=False)
    model_v = UserModel()
    model_v.load_state_dict(torch.load(args.model_1), strict=False)

    [subvolume_num, total_data_number, max_user_id, user_num] = process_data.import_processed_data(head_file_path)
    print("[{}] start test {} processed data from {}".format(datetime.datetime.now(), total_data_number,head_file_path))
    prediction_list = Manager().list()
    test_data = []
    test_data_path_list = []
    for i in range(subvolume_num):
        subvolume_path = "{}.subvolume{}".format(head_file_path, i)
        test_data_path_list.append(subvolume_path)
        if (i+1)%part_data_num==0 or subvolume_num==(i+1):
            task_end = Manager().Queue(1)
            test_task = Process(target=model_test_in_thread,args=[test_data_path_list,[model_t,model_v],batch_size,device,prediction_list,task_end])
            test_task.start()
            test_data_path_list = []
            while task_end.qsize() != 1:
                time.sleep(1)
            test_task.terminate()
    write_submission_file(prediction_list,args.out,"{}_{}_{}-{}_{}_{}".format(
        datetime.datetime.now().year,
        datetime.datetime.now().month,
        datetime.datetime.now().day,
        datetime.datetime.now().hour,
        datetime.datetime.now().minute,
        datetime.datetime.now().second))




