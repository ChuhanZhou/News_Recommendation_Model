from configs.run_config import config as run_config
from configs.model_config import config as model_config
from tool import normalization

import pyarrow.parquet as pq
import pandas as pd
import datetime
import numpy as np
import math

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

def load_dataset(folder_path,type_i=2):
    type_list = ["train","validation","test"]
    path = "{}/{}".format(folder_path,type_list[type_i])
    print("[{}] loading {} dataset from {}".format(datetime.datetime.now(),type_list[type_i],path))

    articles_file = pq.ParquetFile("{}/articles.parquet".format(folder_path))
    articles_data = articles_file.read().to_pandas()

    history_file = pq.ParquetFile("{}/history.parquet".format(path))
    history_data = history_file.read().to_pandas()
    behaviors_file = pq.ParquetFile("{}/behaviors.parquet".format(path))
    behaviors_data = behaviors_file.read().to_pandas()

    article_data = process_articles_data(articles_data)
    user_history_data = process_history_data(history_data)
    train_data,test_data,validation_data = process_behaviors_data(behaviors_data,type_i)


def process_behaviors_data(data,type_i=2):
    user_id_data = list(data["user_id"])
    impression_time_data = list(data["impression_time"])
    article_ids_inview_data = list(data["article_ids_inview"])

    train_data = {}
    test_data = []
    validation_data = []

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

            article_id = article_id_data[i]
            read_time = read_time_data[i]
            scroll_percentage = scroll_percentage_data[i]
            if not math.isnan(article_id) and not math.isnan(read_time) and not math.isnan(scroll_percentage):
                article_id = int(article_id)
                read_time = normalization.read_time_norm(read_time)
                scroll_percentage = scroll_percentage / 100

                user_behavior_np = np.array([article_id, impression_time, read_time, scroll_percentage])
                train_data[user_id].append(user_behavior_np)

            article_id_list = next_article_id_data[i]
            read_time = next_read_time_data[i]
            scroll_percentage = next_scroll_percentage_data[i]

            validation_data.append(article_id_list)

            if len(article_id_list)==1 and not math.isnan(read_time) and not math.isnan(scroll_percentage):
                article_id = int(article_id_list[0])
                read_time = normalization.read_time_norm(read_time)
                scroll_percentage = scroll_percentage / 100

                user_behavior_np = np.array([article_id, impression_time, read_time, scroll_percentage])
                train_data[user_id].append(user_behavior_np)

        test_data.append([user_id,impression_time,inview_list])


    return train_data,test_data,validation_data


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

        category_np = np.zeros((model_config['category_lable_num']))
        category_np[category] = 1
        for subcategory in subcategory_list:
            category_np[subcategory] = 0.5

        sentiment_np = np.zeros((len(model_config['sentiment_label_dict'])))
        sentiment_np[model_config['sentiment_label_dict'][sentiment_label]] = sentiment_score

        article_vector_np = article_vector_data[article_id]

        article_np = np.concatenate((article_type_np, category_np, sentiment_np, article_vector_np), axis=0)

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

        article_data[article_id] = [article_np,history_np]
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
            #impression_time = impression_time
            scroll_percentage = float(scroll_percentage_list[t_i])
            article_id = int(article_id_list[t_i])
            read_time = float(read_time_list[t_i])
            user_history_data[user_id].append([impression_time,article_id,scroll_percentage,read_time])

    return user_history_data