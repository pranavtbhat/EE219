import pandas as pd
import calendar
import re
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(y_actual, y_predicted):
    return  sqrt(mean_squared_error(y_actual, y_predicted))

###
# Utilities for the network backup dataset
###

def network_backup_dataset():
    return pd.read_csv("datasets/network_backup_dataset.csv")

def encode_day_names(days):
    day_to_num = dict(zip(list(calendar.day_name), range(1, 8)))
    return [day_to_num[day] for day in days]

def encode_files(files):
    for i in range(len(files)):
        files[i]=int (files[i].split('_')[-1])
    return files

def encode_work_flows(work_flows):
    for i in range(len(work_flows)):
        work_flows[i]=int (work_flows[i].split('_')[-1])
    return work_flows


