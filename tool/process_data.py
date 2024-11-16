from configs.run_config import config as run_config
from configs.model_config import config as model_config
from tool import normalization

import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
import pandas as pd
import datetime
import numpy as np
import math

import gzip
import pickle
import pickletools
import os
from tqdm import tqdm
import time
from multiprocessing import Manager,Process

def has_file(file_path):
    if file_path == None:
        return False
    return os.path.isfile(file_path)

def load_word2vec_data(file_path=run_config['word2vec_data']):
    print("[{}] loading document vector dataset from {}".format(datetime.datetime.now(), file_path))
    word2vec_data = pq.ParquetFile(file_path).read().to_pandas()
    article_id_data = list(word2vec_data["article_id"])
    document_vector_data = list(word2vec_data["document_vector"])

    article_vector_data = {}
    for i,article_id in enumerate(article_id_data):
        article_vector = document_vector_data[i]
        article_vector_data[article_id] = article_vector
    return article_vector_data

def build_full_running_data(batch_data,category_feature_data,news_info_data):
    [history_feature_list,target_news_feature,label_feature,other_inview_feature_list] = batch_data
    #history_feature: [news_id(1),interest(2),history_timestamp(1),news_history(4)]
    #target_news: [news_id(1),news_history(4)]
    #label: [interest(2)]
    category_dim = []
    news_data_dim = []
    news_history_dim = []
    interest_dim = []
    history_timestamp_dim = []

    target_dim_list = [[],[],[]]#[news_category,news_data,news_history]
    neg_dim_lists = []

    for line_i in range(label_feature.shape[0]):
        for batch_history_feature in history_feature_list:
            history_feature = batch_history_feature[line_i:line_i+1,:]
            news_id = int(history_feature[0,0])
            interest_feature = history_feature[0,1:3]
            history_timestamp = history_feature[0,3]
            news_history_feature = history_feature[0,4:9]

            news_category_feature = np.zeros(model_config['category_label_num'])
            if news_id in category_feature_data:
                news_category_feature = category_feature_data[news_id]
            category_dim.append(torch.tensor(news_category_feature))

            news_data_feature = np.zeros(len(model_config['article_type_dict'])+len(model_config['sentiment_label_dict'])+model_config['news_vector'])
            if news_id in news_info_data:
                news_data_feature = news_info_data[news_id]
            news_data_dim.append(torch.tensor(news_data_feature))
            interest_dim.append(interest_feature)
            history_timestamp_dim.append(history_timestamp)
            news_history_dim.append(news_history_feature)

        target_id = int(target_news_feature[line_i,0])
        target_history = target_news_feature[line_i,1:5]
        target_category = category_feature_data[target_id]
        target_data = news_info_data[target_id]
        target_dim_list[0].append(torch.tensor(target_category))
        target_dim_list[1].append(torch.tensor(target_data))
        target_dim_list[2].append(target_history)

        #neg_dim_list = [[], [], []]
        #for neg_feature in other_inview_feature_list:
        #    neg_id = int(neg_feature[line_i, 0])
        #    neg_history = neg_feature[line_i, 1:5]
        #    if neg_id in category_feature_data:
        #        neg_category = category_feature_data[neg_id]
        #        neg_data = news_info_data[neg_id]
        #        neg_dim_list[0].append(torch.tensor(neg_category))
        #        neg_dim_list[1].append(torch.tensor(neg_data))
        #        neg_dim_list[2].append(neg_history)
        #    else:
        #        neg_dim_list[0].append(torch.zeros(neg_dim_list[0][-1].shape))
        #        neg_dim_list[1].append(torch.zeros(neg_dim_list[1][-1].shape))
        #        neg_dim_list[2].append(torch.zeros(neg_dim_list[2][-1].shape))
        #neg_dim_list[0] = torch.stack(neg_dim_list[0]).float()
        #neg_dim_list[1] = torch.stack(neg_dim_list[1]).float()
        #neg_dim_list[2] = torch.stack(neg_dim_list[2]).float()
        #neg_dim_lists.append(neg_dim_list)

    for neg_feature in other_inview_feature_list:
        neg_dim_list = [[], [], []]
        for line_i in range(neg_feature.shape[0]):
            neg_id = int(neg_feature[line_i, 0])
            neg_history = neg_feature[line_i, 1:5]
            if neg_id in category_feature_data:
                neg_category = category_feature_data[neg_id]
                neg_data = news_info_data[neg_id]
                neg_dim_list[0].append(torch.tensor(neg_category))
                neg_dim_list[1].append(torch.tensor(neg_data))
                neg_dim_list[2].append(neg_history)
            else:
                neg_dim_list[0].append(torch.zeros(neg_dim_lists[-1][0][-1].shape))
                neg_dim_list[1].append(torch.zeros(neg_dim_lists[-1][1][-1].shape))
                neg_dim_list[2].append(torch.zeros(neg_dim_lists[-1][2][-1].shape))
        neg_dim_list[0] = torch.stack(neg_dim_list[0]).float()
        neg_dim_list[1] = torch.stack(neg_dim_list[1]).float()
        neg_dim_list[2] = torch.stack(neg_dim_list[2]).float()
        neg_dim_lists.append(neg_dim_list)

    category_dim = torch.stack(category_dim).float()
    news_data_dim = torch.stack(news_data_dim).float()
    news_history_dim = torch.stack(news_history_dim).float()
    interest_dim = torch.stack(interest_dim).float()
    history_timestamp_dim = torch.stack(history_timestamp_dim).unsqueeze(1).float()

    user_dim_list = [category_dim,news_data_dim,news_history_dim,interest_dim,history_timestamp_dim]

    target_dim_list[0] = torch.stack(target_dim_list[0]).float()
    target_dim_list[1] = torch.stack(target_dim_list[1]).float()
    target_dim_list[2] = torch.stack(target_dim_list[2]).float()

    label_feature = label_feature.float()
    return user_dim_list,target_dim_list,label_feature,neg_dim_lists

def load_processed_dataset(head_file_path,load_data_number=-1):
    [subvolume_num, total_data_number, category_feature_data, news_info_data] = import_processed_data(head_file_path)
    if load_data_number<0:
        load_data_number = total_data_number
    else:
        load_data_number = min(total_data_number,load_data_number)
    print("[{}] start loading {} processed data from {}".format(datetime.datetime.now(),load_data_number,head_file_path))
    time.sleep(0.001)
    processed_data = []
    progress_bar = tqdm(total=load_data_number)
    for i in range(subvolume_num):
        subvolume_path = "{}.subvolume{}".format(head_file_path,i)
        if has_file(subvolume_path):
            part_processed_data = import_processed_data(subvolume_path)
            part_processed_data = part_processed_data[0:min(load_data_number-len(processed_data),len(part_processed_data))]
            processed_data = processed_data + part_processed_data
            progress_bar.update(len(part_processed_data))
            if len(processed_data) >= load_data_number:
                break
    progress_bar.close()
    return processed_data,category_feature_data,news_info_data

def process_dataset(folder_path,type_i=2,subvolume_item_num=30000):
    type_list = ["train","validation","test"]
    path = "{}/{}".format(folder_path,type_list[type_i])
    print("[{}] loading {} dataset from {}".format(datetime.datetime.now(),type_list[type_i],path))

    articles_data = pq.ParquetFile("{}/articles.parquet".format(folder_path)).read().to_pandas()
    history_data = pq.ParquetFile("{}/history.parquet".format(path)).read().to_pandas()
    behaviors_data = pq.ParquetFile("{}/behaviors.parquet".format(path)).read().to_pandas()

    # save memory
    article_data = process_articles_data(articles_data)
    articles_data = None
    print("[{}] articles data processing finished".format(datetime.datetime.now()))
    user_history_data = process_history_data(history_data)
    history_data = None
    print("[{}] history data processing finished".format(datetime.datetime.now()))
    train_part_data, test_part_data, validation_part_data,process_num = process_behaviors_data(behaviors_data, type_i)
    behaviors_data = None
    print("[{}] behaviors data processing finished".format(datetime.datetime.now()))

    print("[{}] start building {} data".format(datetime.datetime.now(),type_list[type_i]))
    processed_data = []
    category_feature_data = {}
    news_info_data = {}

    head_build_task = None
    subvolume_path_list = []
    subvolume_build_task = None

    time.sleep(0.001)
    progress_bar = tqdm(total=process_num)
    data_augmentation_n = 1
    if type_i == 0:
        head_file_path = run_config['train_data_processed']

        for user_i,user_id in enumerate(train_part_data.keys()):
            user_behavior_list = train_part_data[user_id]
            for behavior,other_inview_list in user_behavior_list:
                article_id,standard_time,interest0,interest1 = behavior
                label_feature = np.array([interest0,interest1])

                target_category,target_info,target_history = article_data[article_id]
                target_time_norm = normalization.datetime_norm(target_history[0],standard_time)
                target_news_feature = np.concatenate((np.array([target_time_norm]),target_history[1:4]), axis=0).astype(float)
                target_news_feature = np.concatenate((np.array([article_id]),target_news_feature), axis=0)
                if article_id not in category_feature_data.keys():
                    category_feature_data[article_id] = target_category
                    news_info_data[article_id] = target_info

                other_inview_feature_list = []
                for other_inview_i in range(run_config['neg_label_max_num']):

                    if other_inview_i < len(other_inview_list):
                        other_inview_id = other_inview_list[other_inview_i]
                        other_inview_category, other_inview_info, other_inview_history = article_data[other_inview_id]
                        other_inview_time_norm = normalization.datetime_norm(other_inview_history[0], standard_time)
                        other_inview_feature = np.concatenate((np.array([other_inview_time_norm]), other_inview_history[1:4]), axis=0).astype(float)
                        other_inview_feature = np.concatenate((np.array([other_inview_id]), other_inview_feature),axis=0)
                        other_inview_feature_list.append(other_inview_feature)
                        if other_inview_id not in category_feature_data.keys():
                            category_feature_data[other_inview_id] = other_inview_category
                            news_info_data[other_inview_id] = other_inview_info
                    else:
                        other_inview_feature_list.append(np.zeros(other_inview_feature_list[-1].shape))

                #choose history training data
                user_history_list = user_history_data[user_id].copy()
                user_history_list.reverse()
                # change user_history_list to do Data Augmentation here
                history_choose_list = [user_history_list,user_history_data[user_id].copy()]
                data_augmentation_n = len(history_choose_list)

                for history_choose_area in history_choose_list:
                    train_history_feature_list = []
                    for history_i in range(model_config['history_max_length']):
                        if history_i<len(history_choose_area):
                            user_history = history_choose_area[history_i]
                            history_time, history_news_id, history_interest0, history_interest1 = user_history

                            history_time_norm = normalization.datetime_norm(history_time,standard_time)

                            history_news_category, history_news_info, history_news_history = article_data[history_news_id]
                            history_news_time_norm = normalization.datetime_norm(history_news_history[0], standard_time)
                            history_news_feature = np.concatenate((np.array([history_news_time_norm]), history_news_history[1:4]),axis=0).astype(float)
                            if history_news_id not in category_feature_data.keys():
                                category_feature_data[history_news_id] = history_news_category
                                news_info_data[history_news_id] = history_news_info

                            train_history_feature = np.concatenate((np.array([history_interest0, history_interest1,history_time_norm]),history_news_feature),axis=0)
                            train_history_feature = np.concatenate((np.array([history_news_id]), train_history_feature), axis=0)
                            train_history_feature_list.append(train_history_feature)
                        else:
                            train_history_feature_list.append(np.zeros(train_history_feature_list[-1].shape))
                    processed_data.append([train_history_feature_list,target_news_feature,label_feature,other_inview_feature_list])

                    if len(processed_data) == subvolume_item_num:
                        subvolume_path = "{}.subvolume{}".format(head_file_path,len(subvolume_path_list))
                        next_task = Process(target=export_processed_data,args=[processed_data, subvolume_path])
                        next_task.start()
                        if subvolume_build_task is not None:
                            subvolume_build_task.join()
                            subvolume_build_task.close()
                        subvolume_build_task = next_task
                        subvolume_path_list.append(subvolume_path)
                        processed_data = []
                        if head_build_task is not None:
                            head_build_task.join()
                            head_build_task.close()
                        head_build_task = Process(target=export_processed_data, args=[[len(subvolume_path_list),len(subvolume_path_list)*subvolume_item_num,category_feature_data,news_info_data], head_file_path])
                        head_build_task.start()

                progress_bar.update(1)
            #save memory
            train_part_data[user_id] = None
            user_history_data[user_id] = None
    else:
        head_file_path = run_config['validation_data_processed']

        for test_i,[user_id, standard_time, inview_list] in enumerate(test_part_data):
            test_history_feature_list = []
            user_history_list = user_history_data[user_id].copy()
            user_history_list.reverse()
            for history_i in range(model_config['history_max_length']):
                if history_i < len(user_history_list):
                    user_history = user_history_list[history_i]
                    history_time, history_news_id, history_interest0, history_interest1 = user_history

                    history_time_norm = normalization.datetime_norm(history_time, standard_time)

                    history_news_category, history_news_info, history_news_history = article_data[history_news_id]
                    history_news_time_norm = normalization.datetime_norm(history_news_history[0], standard_time)
                    history_news_feature = np.concatenate((np.array([history_news_time_norm]), history_news_history[1:4]), axis=0).astype(float)
                    if history_news_id not in category_feature_data.keys():
                        category_feature_data[history_news_id] = history_news_category
                        news_info_data[history_news_id] = history_news_info
                    test_history_feature = np.concatenate((np.array([history_interest0, history_interest1, history_time_norm]), history_news_feature),axis=0)
                    test_history_feature = np.concatenate((np.array([history_news_id]), test_history_feature), axis=0)
                    test_history_feature_list.append(test_history_feature)
                else:
                    test_history_feature_list.append(np.zeros(test_history_feature_list[-1].shape))

            inview_news_feature_list = []
            for inview_news_id in inview_list:
                inview_category, inview_info, inview_history = article_data[inview_news_id]
                inview_time_norm = normalization.datetime_norm(inview_history[0], standard_time)
                inview_news_feature = np.concatenate((np.array([inview_time_norm]), inview_history[1:4]),axis=0).astype(float)
                inview_news_feature = np.concatenate((np.array([inview_news_id]), inview_news_feature), axis=0)
                inview_news_feature_list.append(inview_news_feature)
                if inview_news_id not in category_feature_data.keys():
                    category_feature_data[inview_news_id] = inview_category
                    news_info_data[inview_news_id] = inview_info

            part_processed_data = [test_history_feature_list, inview_news_feature_list]

            if type_i == 1:
                label_id = int(validation_part_data[test_i][0])
                part_processed_data.append(label_id)

            processed_data.append(part_processed_data)
            progress_bar.update(1)

            if len(processed_data) == subvolume_item_num:
                subvolume_path = "{}.subvolume{}".format(head_file_path, len(subvolume_path_list))
                next_task = Process(target=export_processed_data, args=[processed_data, subvolume_path])
                next_task.start()
                if subvolume_build_task is not None:
                    subvolume_build_task.join()
                    subvolume_build_task.close()
                subvolume_build_task = next_task
                subvolume_path_list.append(subvolume_path)
                processed_data = []
                if head_build_task is not None:
                    head_build_task.join()
                    head_build_task.close()
                head_build_task = Process(target=export_processed_data, args=[[len(subvolume_path_list), len(subvolume_path_list) * subvolume_item_num, category_feature_data,news_info_data], head_file_path])
                head_build_task.start()

    if head_build_task is not None:
        head_build_task.join()
        head_build_task.close()
    if subvolume_build_task is not None:
        subvolume_build_task.join()
        subvolume_build_task.close()

    subvolume_path = "{}.subvolume{}".format(head_file_path, len(subvolume_path_list))
    export_processed_data(processed_data,subvolume_path)
    subvolume_path_list.append(subvolume_path)
    export_processed_data([len(subvolume_path_list),process_num*data_augmentation_n,category_feature_data,news_info_data], head_file_path)
    progress_bar.close()
    return head_file_path

def process_behaviors_data(data,type_i=2):
    user_id_data = list(data["user_id"])
    impression_time_data = list(data["impression_time"])
    article_ids_inview_data = list(data["article_ids_inview"])

    train_data = {}
    test_data = []
    validation_data = []
    total_num = 0

    if type_i != 2:
        next_article_id_data = list(data["article_ids_clicked"])
        next_read_time_data = list(data["next_read_time"])
        next_scroll_percentage_data = list(data["next_scroll_percentage"])

        article_id_data = list(data["article_id"])
        read_time_data = list(data["read_time"])
        scroll_percentage_data = list(data["scroll_percentage"])

    for i,user_id in enumerate(user_id_data):
        impression_time = np.datetime64(impression_time_data[i])
        inview_list = list(article_ids_inview_data[i])

        if type_i != 2:
            if user_id not in train_data.keys():
                train_data[user_id] = []

            #article_id = article_id_data[i]
            #read_time = read_time_data[i]
            #scroll_percentage = scroll_percentage_data[i]
            #if not math.isnan(article_id) and not math.isnan(read_time) and not math.isnan(scroll_percentage):
            #    article_id = int(article_id)
            #    user_behavior_np = np.array([
            #        article_id,
            #        impression_time,
            #        normalization.read_time_norm(read_time),
            #        scroll_percentage / 100])
            #    train_data[user_id].append(user_behavior_np)
            #    total_num += 1

            article_id_list = next_article_id_data[i]
            read_time = next_read_time_data[i]
            scroll_percentage = next_scroll_percentage_data[i]

            validation_data.append(article_id_list)

            if len(article_id_list)==1 and not math.isnan(read_time) and not math.isnan(scroll_percentage):
                article_id = int(article_id_list[0])

                user_behavior_np = np.array([
                    article_id,
                    impression_time,
                    normalization.read_time_norm(read_time),
                    scroll_percentage/100])

                other_inview_list = []
                for inview_id in inview_list:
                    if inview_id != article_id:
                        other_inview_list.append(inview_id)

                train_data[user_id].append([user_behavior_np,other_inview_list])
                total_num += 1

        test_data.append([user_id,impression_time,inview_list])

    if type_i != 0:
        total_num = len(test_data)

    return train_data,test_data,validation_data,total_num

def process_articles_data(data):
    article_id_data = list(data["article_id"])
    article_type_data = list(data["article_type"])
    category_data = list(data["category"])
    subcategory_data = list(data["subcategory"])
    sentiment_score_data = list(data["sentiment_score"])
    sentiment_label_data = list(data["sentiment_label"])

    #article_history_data
    published_time_data = list(data["published_time"])
    total_inviews_data = list(data["total_inviews"])
    total_pageviews_data = list(data["total_pageviews"])
    total_read_time_data = list(data["total_read_time"])

    article_vector_data = load_word2vec_data()
    article_data = {}

    for i, article_id in enumerate(article_id_data):
        article_id = int(article_id)
        article_type = article_type_data[i]
        category = category_data[i]
        subcategory_list = subcategory_data[i]
        sentiment_score = sentiment_score_data[i]
        sentiment_label = sentiment_label_data[i]

        published_time = np.datetime64(published_time_data[i])
        total_inviews = total_inviews_data[i]
        total_pageviews = total_pageviews_data[i]
        total_read_time = total_read_time_data[i]

        article_type_np = np.zeros((len(model_config['article_type_dict'])))
        article_type_np[model_config['article_type_dict'][article_type]] = 1

        category_np = np.zeros((model_config['category_label_num']))
        category_np[category] = 1
        for subcategory in subcategory_list:
            category_np[subcategory] = 0.5

        sentiment_np = np.zeros((len(model_config['sentiment_label_dict'])))
        sentiment_np[model_config['sentiment_label_dict'][sentiment_label]] = sentiment_score

        article_vector_np = article_vector_data[article_id]

        article_np = np.concatenate((article_type_np, sentiment_np, article_vector_np), axis=0)

        #history data
        if math.isnan(total_inviews):
            total_inviews = 0
        if math.isnan(total_pageviews):
            total_pageviews = 0
        if math.isnan(total_read_time):
            total_read_time = 0

        history_np = np.array([
            published_time,
            normalization.view_num_norm(total_inviews),
            normalization.view_num_norm(total_pageviews),
            normalization.total_read_time_norm(total_read_time)])

        article_data[article_id] = [category_np,article_np,history_np]
    return article_data

def process_history_data(data):
    user_id_data = list(data["user_id"])
    impression_time_data = list(data["impression_time_fixed"])
    scroll_percentage_data = list(data["scroll_percentage_fixed"])
    article_id_data = list(data["article_id_fixed"])
    read_time_data = list(data["read_time_fixed"])

    user_history_data = {}

    for u_i,user_id in enumerate(user_id_data):
        user_id = int(user_id)
        impression_time_list = list(impression_time_data[u_i])
        scroll_percentage_list = list(scroll_percentage_data[u_i])
        article_id_list = list(article_id_data[u_i])
        read_time_list = list(read_time_data[u_i])

        user_history_data[user_id] = []
        for t_i,impression_time in enumerate(impression_time_list):
            scroll_percentage = float(scroll_percentage_list[t_i])
            article_id = int(article_id_list[t_i])
            read_time = float(read_time_list[t_i])
            if math.isnan(scroll_percentage):
                scroll_percentage = 0
            user_history_data[user_id].append([
                impression_time,
                article_id,
                normalization.read_time_norm(read_time),
                scroll_percentage/100])

    return user_history_data

def import_processed_data(path,is_zip=True):
    try:
        if is_zip:
            file = gzip.open(path, 'rb')
            data = pickle.Unpickler(file).load()
        else:
            file = open(path,"rb")
            data = pickle.load(file)
        file.close()
        #print("[{}] import data from {}".format(datetime.datetime.now(), path))
        return data
    except EOFError:
        return None

def export_processed_data(data,path,is_zip=True):
    if is_zip:
        file = gzip.open(path, "wb")
        #file.write(pickletools.optimize(pickle.dumps(data)))
        file.write(pickle.dumps(data))
    else:
        file = open(path, "wb")
        p = pickle.Pickler(file)
        p.fast = True
        p.dump(data)
    file.close()

    #file = open(path,"wb")
    #pickle.dump(data,file,protocol=pickle.HIGHEST_PROTOCOL)
    #file.close()
    #print("[{}] export data to {}".format(datetime.datetime.now(),path))

def build_full_data():
    print()