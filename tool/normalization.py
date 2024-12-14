import math

from configs.model_config import config as model_config
import numpy as np

def read_time_norm(read_time):
    norm = 0
    if not math.isnan(read_time):
        standard = model_config['read_time_normalize_length']*60
        norm = read_time/standard
    return max(0,min(norm,1))

def total_read_time_norm(total_read_time,standard=model_config['total_read_time_norm']):
    norm = 0
    if not math.isnan(total_read_time):
        norm = total_read_time/standard
    return norm

def total_view_num_norm(view_num,standard=model_config['total_views_norm']):
    norm = 0
    if not math.isnan(view_num):
        norm = view_num / standard
    return norm

def value_norm(value, standard):
    norm = 0
    if not math.isnan(value):
        norm = value / standard
    return norm

def sec_norm(total_sec):
    standards=[[60*60*24*365,3000],[60*60*24*30,12],[60*60*24,30],[60*60,23]]
    total_sec = max(0, total_sec)
    norm = []
    sec = total_sec
    for standard,max_num in standards:
        norm.append(min(int(sec/standard),max_num))
        sec-=standard*norm[-1]
    return norm
