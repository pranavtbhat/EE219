import os
import utils
import pandas as pd
import calendar
import matplotlib.pyplot as plt
plt.style.use('ggplot')

if not os.path.exists('plots'):
    os.makedirs('plots')

data = pd.read_csv("datasets/network_backup_dataset.csv")

# Pick days 21 - 40
day_to_num = dict(zip(list(calendar.day_name), range(1, 8)))
data['Day #'] = (data['Week #'] - 1) * 7 + utils.encode_day_names(data['Day of Week'])
first_20_days = data[data['Day #'].map(lambda i : 21 <= i and i <= 40)]

# For each workflow, group by day, sum file sizes and plot
for wid, grp in first_20_days.groupby('Work-Flow-ID'):
    grp.groupby('Day #').sum().plot(
            y='Size of Backup (GB)',
            title = 'File size varaince for ' + wid
    )
    plt.savefig('plots/' + wid + '.png', format='png')
    plt.clf()
