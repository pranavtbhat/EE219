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
for wid, wgrp in first_20_days.groupby('Work-Flow-ID'):
    fig, ax = plt.subplots()
    labels = []

    for fid, fgrp in wgrp.groupby('File Name'):
        ax = fgrp.groupby('Day #').sum().plot(
            ax = ax,
            kind = 'line',
            y = 'Size of Backup (GB)',
        )
        labels.append(fid)

    lines, _ = ax.get_legend_handles_labels()

    plt.xlabel('Time Period (days)')
    plt.ylabel('Data copy size (GB)')

    ax.legend(lines, labels, loc='best')
    ax.set_title('Copy size variance for ' + wid)

    fig.savefig('plots/' + wid + '.png', format='png')

print("Task 1 has been executed")
plt.show()
