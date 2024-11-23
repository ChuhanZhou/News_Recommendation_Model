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

    pca = PCA(n_components=model_config['news_pca_vector'])
    pca_vector_data = pca.fit_transform(np.array(document_vector_data))

    article_vector_data = {}
    for i,article_id in enumerate(article_id_data):
        article_vector = pca_vector_data[i,:]
        article_vector_data[article_id] = article_vector
    return article_vector_data

def load_processed_dataset(head_file_path,load_data_number=-1):
    processed_data = import_processed_data(head_file_path)
    if load_data_number>0:
        return processed_data[0:load_data_number]
    return processed_data

def process_dataset(folder_path,type_i=2):
    type_list = ["train","validation","test"]
    save_file_path = "{}{}_{}".format(run_config['processed_data_path'],folder_path.split("/")[-1],type_list[type_i],)

    path = "{}/{}".format(folder_path,type_list[type_i])
    print("[{}] loading dataset from {}".format(datetime.datetime.now(),path))

    articles_dataset = pq.ParquetFile("{}/articles.parquet".format(folder_path)).read().to_pandas()
    history_dataset = pq.ParquetFile("{}/history.parquet".format(path)).read().to_pandas()
    behaviors_dataset = pq.ParquetFile("{}/behaviors.parquet".format(path)).read().to_pandas()

    # save memory
    article_data = process_articles_data(articles_dataset)
    print("[{}] articles data processing finished".format(datetime.datetime.now()))
    user_history_data = process_history_data(history_dataset)
    print("[{}] history data processing finished".format(datetime.datetime.now()))
    behavior_data = process_behaviors_data(behaviors_dataset, type_i==2)
    print("[{}] behaviors data processing finished".format(datetime.datetime.now()))

    print("[{}] start building processed data".format(datetime.datetime.now()))
    processed_data = []

    time.sleep(0.001)
    progress_bar = tqdm(total=len(behavior_data))

    for behavior_info in behavior_data:
        [user_id,inview_list,target] = behavior_info
        user_history_list = user_history_data[user_id][0:model_config['history_max_length']]

        full_history_data_list = []

        for history in user_history_list:
            [article_id, read_time, scroll_percentage] = history
            [article_vector_np,category,subcategory_list,sentiment_np,article_type_i] = article_data[article_id]
            full_history_data_list.append([article_vector_np,category,subcategory_list,sentiment_np,article_type_i,read_time, scroll_percentage])

        full_inview_data_list = []
        label_list = []
        for inview_id in inview_list:
            if inview_id == target:
                label_list.append(1)
            else:
                label_list.append(0)

            [article_vector_np, category, subcategory_list, sentiment_np, article_type_i] = article_data[inview_id]
            full_inview_data_list.append([article_vector_np, category, subcategory_list, sentiment_np, article_type_i])

        processed_data.append([user_id,full_history_data_list,full_inview_data_list,label_list])
        progress_bar.update(1)
    progress_bar.close()

    export_processed_data(processed_data, save_file_path)
    return processed_data

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

    article_vector_data = load_word2vec_data()
    article_data = {}

    for i, article_id in enumerate(article_id_data):
        article_id = int(article_id)
        article_type = article_type_data[i]
        category = category_data[i]
        subcategory_list = subcategory_data[i]
        sentiment_score = sentiment_score_data[i]
        sentiment_label = sentiment_label_data[i]

        article_vector_np = article_vector_data[article_id]
        article_type_i = model_config['article_type_dict'][article_type]

        sentiment_np = np.zeros((len(model_config['sentiment_label_dict'])))
        sentiment_np[model_config['sentiment_label_dict'][sentiment_label]] = sentiment_score

        article_data[article_id] = [article_vector_np,category,subcategory_list,sentiment_np,article_type_i]
    return article_data

def process_history_data(data):
    user_id_data = list(data["user_id"])
    #impression_time_data = list(data["impression_time_fixed"])
    scroll_percentage_data = list(data["scroll_percentage_fixed"])
    article_id_data = list(data["article_id_fixed"])
    read_time_data = list(data["read_time_fixed"])

    user_history_data = {}

    for u_i,user_id in enumerate(user_id_data):
        user_id = int(user_id)
        #impression_time_list = list(impression_time_data[u_i])
        scroll_percentage_list = list(scroll_percentage_data[u_i]).copy()
        article_id_list = list(article_id_data[u_i]).copy()
        read_time_list = list(read_time_data[u_i]).copy()

        scroll_percentage_list.reverse()
        article_id_list.reverse()
        read_time_list.reverse()

        user_history_data[user_id] = []
        for i,article_id in enumerate(article_id_list):
            read_time = float(read_time_list[i])
            scroll_percentage = float(scroll_percentage_list[i])
            if math.isnan(scroll_percentage):
                scroll_percentage = 0
            user_history_data[user_id].append([article_id,read_time,scroll_percentage])
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