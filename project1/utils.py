import pandas as pd
import calendar
import re
import numpy as np

def rmse(a1, a2):
    return np.sqrt(np.mean((a1 - a2) ** 2))

###
# Utilities for the network backup dataset
###

def network_backup_dataset():
    return pd.read_csv("datasets/network_backup_dataset.csv")

def encode_day_names(days):
    day_to_num = dict(zip(list(calendar.day_name), range(1, 8)))
    return [day_to_num[day] for day in days]

def encode_files(files):
    return [re.match(r'File_((\d))', f).group(1) for f in files]

def encode_work_flows(work_flows):
    return [re.match(r'work_flow_((\d))', wf).group(1) for wf in work_flows]


