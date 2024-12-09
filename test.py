import queue

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
import copy

def model_test_in_thread(test_data_path_list,model_list,batch_size,device,prediction_queue,id_list,status):
    test_data = []
    for path in test_data_path_list:
        part_test_data = process_data.import_processed_data(path)
        print("load {} data from {}".format(len(part_test_data), path))
        test_data+=part_test_data
    model_test(model_list,test_data,device,prediction_queue,id_list,batch_size)
    status.put(0)

@torch.no_grad()
def model_test(model_list,test_data,device=run_config['device'],prediction_queue=None,id_list=None,batch_size=1):
    test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    for model in model_list:
        model.eval()
        model.to(device)

    if prediction_queue is None:
        prediction_queue = queue.Queue()
    if id_list is None:
        id_list = []
    time.sleep(0.001)
    progress_bar = tqdm(desc="[{}] testing model".format(datetime.datetime.now()), total=len(test_data))
    softmax = nn.Softmax(dim=1)
    for data in test_data_loader:
        [impression_id,user_id, x_history, x_inview, x_global, _, label_id, empty_num] = data
        min_zero_data_num = torch.min(empty_num)
        x_history = x_history.to(device)
        x_inview = x_inview.to(device)
        x_global = x_global.to(device)
        if min_zero_data_num>0:
            x_inview = x_inview[:,0:-min_zero_data_num]
            x_global = x_global[:,0:-min_zero_data_num]
            label_id = label_id[:,0:-min_zero_data_num]
            empty_num = empty_num-min_zero_data_num

        out = None
        for model in model_list:
            if out is None:
                out = softmax(model(x_history, x_inview, x_global))
            else:
                out += softmax(model(x_history, x_inview, x_global))
        out = out/len(model_list)
        for d_i in range(out.shape[0]):
            zero_data_num = int(empty_num[d_i])
            if zero_data_num>0:
                prediction_queue.put([impression_id[d_i],user_id[d_i],softmax(out[d_i:d_i+1,0:-zero_data_num]).squeeze(0).cpu().numpy(),label_id[d_i,0:-zero_data_num].numpy()])
            else:
                prediction_queue.put([impression_id[d_i],user_id[d_i],out[d_i].cpu().numpy(),label_id[d_i].numpy()])
            id_list.append("{}_{}".format(int(impression_id[d_i]),int(user_id[d_i])))
            progress_bar.update()
    progress_bar.close()
    return prediction_queue,id_list

def write_submission_file(prediction_queue,id_list,path=run_config['output_path'],name="predictions",thread_num=run_config['thread_num']):
    file_path = path + "predictions.txt"
    zip_path = path + "{}.zip".format(name)

    process_task_list = []
    processed_string_dict = Manager().dict()
    finish_queue = Manager().Queue(thread_num)
    time.sleep(0.001)
    progress_bar = tqdm(total=len(id_list),desc="[{}] predict result to string".format(datetime.datetime.now()))
    for i in range(thread_num):
        process_task = Process(target=get_string_of_prediction,args=[prediction_queue, processed_string_dict,finish_queue])
        process_task.start()
        process_task_list.append(process_task)

    total_process = 0
    while not prediction_queue.empty():
        time.sleep(0.2)
        now_process = len(processed_string_dict)
        progress_bar.update(now_process - total_process)
        progress_bar.set_postfix(running_threads="{}/{}".format(len(process_task_list) - finish_queue.qsize(), len(process_task_list)))
        total_process = now_process

    for process_task in process_task_list:
        process_task.terminate()
    progress_bar.set_postfix(running_threads="{}/{}".format(0, len(process_task_list)))
    progress_bar.close()

    time.sleep(0.001)
    progress_bar = tqdm(desc="[{}] writing string prediction".format(datetime.datetime.now()), total=len(id_list))

    with open(file_path, "w", encoding="utf-8") as file:
        for id in id_list:
            part_submit_info = processed_string_dict[id]
            file.write(part_submit_info)
            progress_bar.update()
    progress_bar.close()

    zip_file = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)
    zip_file.write(file_path,arcname=file_path.split("/")[-1])
    zip_file.close()
    return zip_path

def get_string_of_prediction(prediction_queue,return_dict,finish_queue):
    finish_queue.put(0)
    while not prediction_queue.empty():
        impression_id,user_id,prediction_scores,ids = prediction_queue.get()
        finish_queue.get(0)

        sortrd_result = sorted(enumerate(prediction_scores), key=lambda x: x[1], reverse=True)
        rank_result = ["-1"] * len(prediction_scores)
        for r_i, [i, v] in enumerate(sortrd_result):
            rank_result[i] = str(r_i + 1)

        result_info = ",".join(rank_result)
        submit_info = "{} [{}]\n".format(int(impression_id), result_info)
        return_dict["{}_{}".format(int(impression_id),int(user_id))] = submit_info
        finish_queue.put(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test model")
    parser.add_argument('--data', help="test dataset path", default=run_config['processed_data_path'] + run_config['test_data_processed'])
    parser.add_argument('--volume', help="how many data volumes to read together", type=int, default=10)
    parser.add_argument('--batch', help="batch size", type=int, default=80)
    parser.add_argument('--out', help="result output path", default=run_config['output_path'])
    parser.add_argument('--model_0', help="model parameter file 0", default=None)
    parser.add_argument('--model_1', help="model parameter file 1", default=None)
    parser.add_argument('--thread', help="how may thread can be used", type=int, default=run_config['thread_num'])
    args = parser.parse_args()

    run_config['thread_num'] = args.thread
    head_file_path = args.data
    part_data_num = args.volume
    batch_size = args.batch
    device = run_config['device']
    if args.model_0 is None and args.model_1 is None:
        args.model_0 = run_config['ckpt_save_path'] + "ckpt_ebnerd_large_train_final.pth"
        args.model_1 = run_config['ckpt_save_path'] + "ckpt_ebnerd_large_validation_final.pth"

    print("[{}] device: {}".format(datetime.datetime.now(), device))
    model_list = []
    for model_path in [args.model_0,args.model_1]:
        if model_path is not None:
            print("[{}] load model parameter from {}".format(datetime.datetime.now(),model_path))
            model = UserModel()
            model.load_state_dict(torch.load(model_path), strict=False)
            model_list.append(model)
    #model_t = UserModel()
    #model_t.load_state_dict(torch.load(args.model_0), strict=False)
    #model_v = UserModel()
    #model_v.load_state_dict(torch.load(args.model_1), strict=False)

    [subvolume_num, total_data_number, max_user_id, user_num] = process_data.import_processed_data(head_file_path)
    print("[{}] start test {} processed data from {}".format(datetime.datetime.now(), total_data_number,head_file_path))
    prediction_queue = Manager().Queue(total_data_number)
    id_list = Manager().list()
    test_data = []
    test_data_path_list = []
    for i in range(subvolume_num):
        subvolume_path = "{}.subvolume{}".format(head_file_path, i)
        test_data_path_list.append(subvolume_path)
        if (i+1)%part_data_num==0 or subvolume_num==(i+1):
            task_end = Manager().Queue(1)
            test_task = Process(target=model_test_in_thread,args=[test_data_path_list,model_list,batch_size,device,prediction_queue,id_list,task_end])
            test_task.start()
            test_data_path_list = []
            while task_end.qsize() != 1:
                time.sleep(1)
            test_task.terminate()
    zip_path = write_submission_file(prediction_queue,id_list,args.out,"{}_{}_{}-{}_{}_{}".format(
        datetime.datetime.now().year,
        datetime.datetime.now().month,
        datetime.datetime.now().day,
        datetime.datetime.now().hour,
        datetime.datetime.now().minute,
        datetime.datetime.now().second))
    print("[{}] save result in {}".format(datetime.datetime.now(),zip_path))




