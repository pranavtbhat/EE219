from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
import statsmodels.formula.api as sm
import Functions

from OneHotEncode import one_hot_dataframe
import Plots

import utils

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


### Changed the column names in data file to use formula in ols
# Load the dataset
# Columns:
# 0 -> WeekNo
# 1 -> DayofWeek
# 2 -> BackupStartTime
# 3 -> WorkFlowID
# 4 -> FileName
# 5 -> SizeofBackupGB
# 6 -> BackupTimeHour
###
data = pd.read_csv("datasets/network_backup_dataset.csv")
data = data.replace({'DayofWeek': {'Monday' : 0, 'Tuesday' : 1, 'Wednesday' : 2 , 'Thursday' : 3, 'Friday' : 4,
                                                     'Saturday' : 5, 'Sunday' : 6 }})


X = data.ix[:, [0, 1, 2, 3,4, 6]].values
X[:, 3] = utils.encode_work_flows(X[:, 3])
X[:, 4] = utils.encode_files(X[:, 4])

y = data.ix[:, 5].values


uniqueWorkFlow = sorted(pd.unique(data['WorkFlowID']))  # get unique workFlow values
uniqueFiles = ['File_{0}'.format(s) for s in xrange(len((pd.unique(data['FileName']))))]   # get unique fileName values

networkDataset=data
for i,j in zip(uniqueWorkFlow,range(len(uniqueWorkFlow))):
    networkDataset = networkDataset.replace({'WorkFlowID': {i : j}})

for i,j in zip(uniqueFiles,range(len(uniqueFiles))):
    networkDataset = networkDataset.replace({'FileName': {i : j}})

## Uncomment to use data without columns FileName and WeekNumber
# dataNew=data
# dataNew['WorkFlowID']=networkDataset['WorkFlowID']
# dataNew['FileName']=networkDataset['FileName']
# dataNew['SizeofBackupGB']=y
# del dataNew['FileName']
# del dataNew['WeekNo']

###
# Part a: Linear Regression Model
# Check the graphs
###
reg = linear_model.LinearRegression()

y_predicted = cross_val_predict(reg, X, y, cv=10)
print utils.rmse(y,y_predicted)

#Almost same as above rmse
cv_scores = cross_val_score(reg, X, y, cv=10, scoring='neg_mean_squared_error')
print (sum(cv_scores)/-10.0)**0.5

fig, ax = plt.subplots()
ax.scatter(x=y, y=y_predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()],  'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Fitted')
plt.savefig('plots/lrActualvsFitted-4features.png', format='png')
plt.clf()


#Residual
y_residual = y - y_predicted

fig, ax = plt.subplots()
ax.scatter(y_predicted, y_residual)
ax.set_xlabel('Fitted')
ax.set_ylabel('Residual')
plt.savefig('plots/lrFittedvsResidual.png', format='png')
plt.clf()


model = sm.ols('SizeofBackupGB ~ WeekNo+ DayofWeek + BackupStartTime + WorkFlowID+FileName + BackupTimeHour ',networkDataset).fit()
print model.summary()

## Uncomment to check the effect with one hot encoding
# network_data = pd.read_csv('datasets/network_backup_dataset.csv')
# #One Hot Encoding
# one_hot_data, _, _ = one_hot_dataframe(network_data, ['DayofWeek', 'WorkFlowID','FileName'], replace=True)
#
# feature_cols = [col for col in one_hot_data.columns if col not in ['SizeofBackupGB']]
# X = one_hot_data[feature_cols]
# y = one_hot_data['SizeofBackupGB']
#
# all_columns = " + ".join(one_hot_data.columns - ["SizeofBackupGB"])
#
# my_formula = "'SizeofBackupGB ~ " + all_columns+"'"
# print my_formula
#
# model = sm.ols(formula=my_formula,data=one_hot_data).fit()
# print model.summary()
