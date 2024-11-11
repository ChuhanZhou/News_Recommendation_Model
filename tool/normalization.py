from configs.model_config import config as model_config
import numpy as np

def read_time_norm(read_time):
    standard = model_config['read_time_normalize_length']*60
    norm = read_time/standard
    return max(0,min(norm,1))

def total_read_time_norm(total_read_time):
    standard = model_config['total_inview_normalize_length']*model_config['read_time_normalize_length']*60
    norm = total_read_time/standard
    return max(0,min(norm,1))

def view_num_norm(view_num):
    standard = model_config['total_inview_normalize_length']
    norm = view_num / standard
    return max(0, min(norm, 1))

def datetime_norm(datetime, standard_time):
    standard = model_config['datetime_normalize_length']
    a = np.timedelta64(standard, 'D')
    b = standard_time - datetime
    norm = (standard_time - datetime)/np.timedelta64(standard, 'D')
    return 1-max(0, min(norm, 1))