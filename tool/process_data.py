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
from sklearn.decomposition import PCA
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

    pca = PCA(n_components=model_config['text_pca_vector'])
    pca_vector_data = pca.fit_transform(np.array(document_vector_data))

    text_vector_data = {}
    for i,article_id in enumerate(article_id_data):
        text_vector = pca_vector_data[i,:]
        text_vector_data[int(article_id)] = text_vector
    return text_vector_data

def load_image_embeddings_data(file_path=run_config['image_embeddings_data']):
    print("[{}] loading image embeddings dataset from {}".format(datetime.datetime.now(), file_path))
    image_embeddings_data = pq.ParquetFile(file_path).read().to_pandas()
    article_id_data = list(image_embeddings_data["article_id"])
    img_vector_data = list(image_embeddings_data["image_embedding"])

    pca = PCA(n_components=model_config['img_pca_vector'])
    pca_vector_data = pca.fit_transform(np.array(img_vector_data))

    img_vector_data = {}
    for i,article_id in enumerate(article_id_data):
        img_vector = pca_vector_data[i,:]
        img_vector_data[int(article_id)] = img_vector
    return img_vector_data

def load_text_img_data(text_file_path=run_config['word2vec_data'],img_file_path=run_config['image_embeddings_data']):
    print("[{}] loading document vector dataset from {}".format(datetime.datetime.now(), text_file_path))
    word2vec_data = pq.ParquetFile(text_file_path).read().to_pandas()
    text_article_id_data = list(word2vec_data["article_id"])
    text_data = list(word2vec_data["document_vector"])
    print("[{}] loading image embeddings dataset from {}".format(datetime.datetime.now(), img_file_path))
    image_embeddings_data = pq.ParquetFile(img_file_path).read().to_pandas()
    img_article_id_data = list(image_embeddings_data["article_id"])
    img_data = list(image_embeddings_data["image_embedding"])

    img_vector_data = {}
    for i, article_id in enumerate(img_article_id_data):
        img_vector_data[int(article_id)] = img_data[i]

    text_img_data = []
    for i, article_id in enumerate(text_article_id_data):
        text_vector = text_data[i]
        if article_id in img_vector_data:
            img_vector = img_vector_data[article_id]
        else:
            img_vector = np.zeros(img_data[0].shape)
        text_img_data.append(np.append(text_vector,img_vector))
    text_img_data = np.array(text_img_data)

    pca = PCA(n_components=model_config['pca_vector'])
    pca_vector_data = pca.fit_transform(np.array(text_img_data))

    text_img_vector_data = {}
    for i, article_id in enumerate(text_article_id_data):
        text_img_vector = pca_vector_data[i, :]
        text_img_vector_data[int(article_id)] = text_img_vector
    return text_img_vector_data

def load_processed_dataset(head_file_path,load_data_number=-1):
    [subvolume_num, total_data_number,max_user_id] = import_processed_data(head_file_path)
    if load_data_number < 0:
        load_data_number = total_data_number
    else:
        load_data_number = min(total_data_number, load_data_number)
    print("[{}] start loading {} processed data from {}".format(datetime.datetime.now(), load_data_number,head_file_path))
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
    return processed_data,max_user_id

def process_dataset(folder_path,type_i=2,subvolume_item_num=30000,for_batch=True):
    type_list = ["train","validation","test"]
    save_file_path = "{}{}_{}".format(run_config['processed_data_path'],folder_path.split("/")[-1],type_list[type_i])
    if for_batch:
        save_file_path = "{}_batch".format(save_file_path)

    path = "{}/{}".format(folder_path,type_list[type_i])
    print("[{}] loading dataset from {}".format(datetime.datetime.now(),path))

    articles_dataset = pq.ParquetFile("{}/articles.parquet".format(folder_path)).read().to_pandas()
    history_dataset = pq.ParquetFile("{}/history.parquet".format(path)).read().to_pandas()
    behaviors_dataset = pq.ParquetFile("{}/behaviors.parquet".format(path)).read().to_pandas()

    # save memory
    article_data = process_articles_data(articles_dataset)
    articles_dataset = None
    print("[{}] articles data processing finished".format(datetime.datetime.now()))
    user_history_data = process_history_data(history_dataset)
    history_dataset = None
    print("[{}] history data processing finished".format(datetime.datetime.now()))
    behavior_data = process_behaviors_data(behaviors_dataset, type_i==2)
    behaviors_dataset = None
    print("[{}] behaviors data processing finished".format(datetime.datetime.now()))

    print("[{}] start building processed data".format(datetime.datetime.now()))
    processed_data = []

    head_build_task = None
    subvolume_path_list = []
    subvolume_build_task = None

    time.sleep(0.001)
    progress_bar = tqdm(total=len(behavior_data))
    max_user_id = 0

    for behavior_info in behavior_data:
        [user_id,inview_list,target] = behavior_info
        max_user_id = max(max_user_id,user_id)
        user_history_list = user_history_data[user_id][0:model_config['history_max_num']]

        full_history_data_list = []

        for h_i in range(model_config['history_max_num']):
            if h_i < len(user_history_list):
                [time_np, article_id, read_time, scroll_percentage] = user_history_list[h_i]
                [text_img_vector_np, category, subcategory_np, sentiment_np, article_type_i,_,_,_,_] = article_data[article_id]
                history_np = np.concatenate((time_np,text_img_vector_np,np.array([category]),subcategory_np,sentiment_np,np.array([article_type_i,read_time,scroll_percentage])),axis=0)
                full_history_data_list.append(history_np)
            else:
                if for_batch:
                    full_history_data_list.append(np.zeros(full_history_data_list[-1].shape))
                else:
                    break

        full_inview_data_list = []
        global_inview_data_list = []
        label_true_list = []
        label_id_list = []
        for inview_id in inview_list:
            if for_batch and len(label_true_list) == model_config['inview_max_num']-1 and sum(label_true_list) == 0:
                if inview_id == target:
                    label_true_list.append(1)
                    label_id_list.append(inview_id)
                    [text_img_vector_np, category, subcategory_np, sentiment_np, article_type_i,time_np,total_inviews,total_pageviews,total_read_time] = article_data[inview_id]
                    inview_np = np.concatenate((time_np,text_img_vector_np, np.array([category]), subcategory_np, sentiment_np,np.array([article_type_i])), axis=0)
                    full_inview_data_list.append(inview_np)
                    global_inview_data_list.append(np.array([total_inviews,total_pageviews,total_read_time]))
            else:
                if inview_id == target:
                    label_true_list.append(1)
                else:
                    label_true_list.append(0)
                label_id_list.append(inview_id)

                [text_img_vector_np, category, subcategory_np, sentiment_np, article_type_i,time_np,total_inviews,total_pageviews,total_read_time] = article_data[inview_id]
                inview_np = np.concatenate((time_np,text_img_vector_np,np.array([category]),subcategory_np,sentiment_np,np.array([article_type_i])),axis=0)
                full_inview_data_list.append(inview_np)
                global_inview_data_list.append(np.array([total_inviews, total_pageviews, total_read_time]))

            if for_batch and len(label_true_list) >= model_config['inview_max_num']:
                break

        if for_batch:
            for i in range(model_config['inview_max_num']-len(label_true_list)):
                label_true_list.append(0)
                label_id_list.append(-1)
                full_inview_data_list.append(np.zeros(full_inview_data_list[-1].shape))
                global_inview_data_list.append(np.zeros(global_inview_data_list[-1].shape))

        full_history_data_list = np.array(full_history_data_list)
        full_inview_data_list = np.array(full_inview_data_list)
        global_inview_data_list = np.array(global_inview_data_list)
        label_true_list = np.array(label_true_list)
        label_id_list = np.array(label_id_list)

        processed_data.append([user_id,full_history_data_list,full_inview_data_list,global_inview_data_list,label_true_list,label_id_list])
        progress_bar.update(1)

        if len(processed_data) == subvolume_item_num:
            subvolume_path = "{}.subvolume{}".format(save_file_path, len(subvolume_path_list))
            if subvolume_build_task is not None:
                subvolume_build_task.join()
                subvolume_build_task.close()
            subvolume_build_task = Process(target=export_processed_data, args=[processed_data, subvolume_path])
            subvolume_build_task.start()
            subvolume_path_list.append(subvolume_path)
            processed_data = []
            if head_build_task is not None:
                head_build_task.join()
                head_build_task.close()
            head_build_task = Process(target=export_processed_data, args=[[len(subvolume_path_list), len(subvolume_path_list) * subvolume_item_num, max_user_id], save_file_path])
            head_build_task.start()

            if len(subvolume_path_list)==21:
                break

    progress_bar.close()

    if head_build_task is not None:
        head_build_task.join()
        head_build_task.close()
    if subvolume_build_task is not None:
        subvolume_build_task.join()
        subvolume_build_task.close()

    subvolume_path = "{}.subvolume{}".format(save_file_path, len(subvolume_path_list))
    export_processed_data(processed_data, subvolume_path)
    subvolume_path_list.append(subvolume_path)
    export_processed_data([len(subvolume_path_list), len(behavior_data),max_user_id],save_file_path)
    return save_file_path

def process_behaviors_data(data,is_test=False):
    user_id_data = list(data["user_id"])
    #impression_time_data = list(data["impression_time"])
    article_ids_inview_data = list(data["article_ids_inview"])

    behavior_data = []

    if not is_test:
        next_article_id_data = list(data["article_ids_clicked"])

    for i,user_id in enumerate(user_id_data):
        inview_list = list(article_ids_inview_data[i])
        if not is_test:
            article_id_list = next_article_id_data[i]
            if len(article_id_list) == 1:
                article_id = int(article_id_list[0])
                behavior_data.append([user_id,inview_list,article_id])
        else:
            behavior_data.append([user_id,inview_list,None])
    return behavior_data

def process_articles_data(data):
    article_id_data = list(data["article_id"])
    article_type_data = list(data["article_type"])
    category_data = list(data["category"])
    subcategory_data = list(data["subcategory"])
    sentiment_score_data = list(data["sentiment_score"])
    sentiment_label_data = list(data["sentiment_label"])

    published_time_data = list(data["published_time"])
    total_inviews_data = list(data["total_inviews"])
    total_pageviews_data = list(data["total_pageviews"])
    total_read_time_data = list(data["total_read_time"])

    text_img_vector_data = load_text_img_data()

    article_data = {}
    for i, article_id in enumerate(article_id_data):
        article_id = int(article_id)
        article_type = article_type_data[i]
        category = category_data[i]
        subcategory_list = subcategory_data[i]
        sentiment_score = sentiment_score_data[i]
        sentiment_label = sentiment_label_data[i]

        published_time = [published_time_data[i].year,published_time_data[i].month,published_time_data[i].day,published_time_data[i].hour]
        global_info_list = [total_inviews_data[i],total_pageviews_data[i],total_read_time_data[i]]
        norm_standard_list = [model_config['total_views_norm'],model_config['total_views_norm'],model_config['total_read_time_norm']]
        for v_i,v in enumerate(global_info_list):
            global_info_list[v_i] = normalization.value_norm(global_info_list[v_i],norm_standard_list[v_i])
        total_inviews,total_pageviews,total_read_time = global_info_list

        text_img_vector_np = text_img_vector_data[article_id]

        subcategory_label_list = []
        for sc_i in range(model_config['subcategory_max_num']):
            if sc_i < len(subcategory_list):
                subcategory_label_list.append(subcategory_list[sc_i])
            else:
                subcategory_label_list.append(0)

        article_type_i = model_config['article_type_dict'][article_type]

        sentiment_np = np.zeros((len(model_config['sentiment_label_dict'])))
        sentiment_np[model_config['sentiment_label_dict'][sentiment_label]] = sentiment_score

        article_data[article_id] = [text_img_vector_np,category,np.array(subcategory_label_list),sentiment_np,article_type_i,np.array(published_time),total_inviews,total_pageviews,total_read_time]
    return article_data

def process_history_data(data):
    user_id_data = list(data["user_id"])
    scroll_percentage_data = list(data["scroll_percentage_fixed"])
    article_id_data = list(data["article_id_fixed"])
    read_time_data = list(data["read_time_fixed"])

    impression_time_data = list(data["impression_time_fixed"])

    user_history_data = {}
    for u_i,user_id in enumerate(user_id_data):
        user_id = int(user_id)
        scroll_percentage_list = list(scroll_percentage_data[u_i])
        article_id_list = list(article_id_data[u_i])
        read_time_list = list(read_time_data[u_i])

        impression_time_list = list(impression_time_data[u_i])

        scroll_percentage_list.reverse()
        article_id_list.reverse()
        read_time_list.reverse()
        impression_time_list.reverse()

        user_history_data[user_id] = []
        for i,article_id in enumerate(article_id_list):
            read_time = normalization.value_norm(float(read_time_list[i]),model_config['read_time_norm'])
            scroll_percentage = normalization.value_norm(float(scroll_percentage_list[i]),model_config['scroll_norm'])
            impression_time = pd.to_datetime(impression_time_list[i])
            impression_time = [impression_time.year, impression_time.month, impression_time.day,impression_time.hour]
            #if math.isnan(scroll_percentage):
            #    scroll_percentage = 0
            user_history_data[user_id].append([np.array(impression_time),article_id,read_time,scroll_percentage])
            if (i+1)>=model_config['history_max_num']:
                break
    return user_history_data

def import_processed_data(path):
    try:
        file = gzip.open(path, 'rb')
        data = pickle.Unpickler(file).load()
        file.close()
        return data
    except EOFError:
        return None

def export_processed_data(data,path):
    file = gzip.open(path, "wb")
    file.write(pickle.dumps(data))
    file.close()