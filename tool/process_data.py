from configs.run_config import config as run_config
from configs.model_config import config as model_config
from tool import normalization

import pyarrow.parquet as pq
import pandas as pd
import datetime
import numpy as np
import math
import pickle

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
    train_part_data,test_part_data,validation_part_data = process_behaviors_data(behaviors_data,type_i)

    print("[{}] start building {} data".format(datetime.datetime.now(),type_list[type_i]))
    processed_data = []
    category_feature_data = {}
    news_info_data = {}
    if type_i == 0:
        for user_i,user_id in enumerate(train_part_data.keys()):
            user_behavior_list = train_part_data[user_id]
            for behavior in user_behavior_list:
                article_id,standard_time,interest0,interest1 = behavior
                label_feature = np.array([interest0,interest1])

                target_category,target_info,target_history = article_data[article_id]
                target_time_norm = normalization.datetime_norm(target_history[0],standard_time)
                target_news_feature = np.concatenate((np.array([target_time_norm]),target_history[1:4]), axis=0).astype(float)
                target_news_feature = np.concatenate((np.array([article_id]),target_news_feature), axis=0)
                category_feature_data[article_id] = target_category
                news_info_data[article_id] = target_info

                #choose history training data
                train_history_feature_list = []
                #change user_history_list to do Data Augmentation
                user_history_list = user_history_data[user_id].copy()
                user_history_list.reverse()

                for history_i in range(model_config['history_max_length']):
                    if history_i<len(user_history_list):
                        user_history = user_history_list[history_i]
                        history_time, history_news_id, history_interest0, history_interest1 = user_history
                        history_time_norm = normalization.datetime_norm(history_time,standard_time)

                        history_news_category, history_news_info, history_news_history = article_data[history_news_id]
                        history_news_time_norm = normalization.datetime_norm(history_news_history[0], standard_time)
                        history_news_feature = np.concatenate((np.array([history_news_time_norm]), history_news_history[1:4]),axis=0).astype(float)
                        category_feature_data[history_news_id] = history_news_category
                        news_info_data[history_news_id] = history_news_info

                        train_history_feature = np.concatenate((np.array([history_interest0, history_interest1,history_time_norm]),history_news_feature),axis=0)
                        train_history_feature = np.concatenate((np.array([history_news_id]), train_history_feature), axis=0)
                        train_history_feature_list.append(train_history_feature)
                    else:
                        train_history_feature_list.append(np.zeros(train_history_feature_list[-1].shape))
                processed_data.append([train_history_feature_list,target_news_feature,label_feature])

            #schedule log
            if (user_i+1)%int(len(train_part_data)/10)==0 or user_i+1==len(train_part_data):
                print("[{}] training data building: ({}/{})".format(datetime.datetime.now(),user_i+1,len(train_part_data)))
    elif type_i == 1:
        a=1
    elif type_i == 2:
        a=1
    return processed_data,category_feature_data,news_info_data

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

                user_behavior_np = np.array([
                    article_id,
                    impression_time,
                    normalization.read_time_norm(read_time),
                    scroll_percentage/100])
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
            user_history_data[user_id].append([
                impression_time,
                article_id,
                normalization.read_time_norm(read_time),
                scroll_percentage/100])

    return user_history_data

def import_processed_data(path):
    try:
        file = open(path,"rb")
        data = pickle.load(file)
        file.close()
        print("[{}] import data from {}".format(datetime.datetime.now(), path))
        return data
    except EOFError:
        return None

def export_processed_data(data,path):
    file = open(path,"wb")
    pickle.dump(data,file)
    file.close()
    print("[{}] export data to {}".format(datetime.datetime.now(),path))

def build_full_data():
    print()