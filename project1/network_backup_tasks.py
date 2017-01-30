import os
import pandas as pd
import calendar
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

if not os.path.exists('plots'):
    os.makedirs('plots')

data = pd.read_csv("datasets/network_backup_dataset.csv")

day_to_num = dict(zip(list(calendar.day_name), range(1, 8)))
print day_to_num
data['Day #'] = (data['Week #'] - 1) * 7 + [day_to_num[day] for day in data['Day of Week']]
data['DayInTP'] = (data['Day #']-1)%21;
data['TimePeriod #'] = (data['Day #']-1)//21;
print max(data['Week #'])
print max(data['Day #'])
print max(data['TimePeriod #'])
for wid in ('work_flow_0','work_flow_1','work_flow_2','work_flow_3','work_flow_4'):
    print wid
    workflow_id = data['Work-Flow-ID'] == wid
    data_wifacebookd = data[workflow_id]
    for tp in range(min(data['TimePeriod #']),max(data['TimePeriod #'])+1,1):
        first_21_days = data_wid[data_wid['TimePeriod #']==tp]
        x=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        y=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for dayInTP in range(0,21,1):
            x[dayInTP]= dayInTP
            y[dayInTP]= sum(first_21_days[first_21_days['DayInTP']==dayInTP]['Size of Backup (GB)'])
        plt.plot(x,y)
    plt.savefig('plots/' + wid + '.png', format='png')
    plt.clf()
