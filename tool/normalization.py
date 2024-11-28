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