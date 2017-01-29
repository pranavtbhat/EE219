from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
import statsmodels.formula.api as sm

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
kf = KFold(n_splits = 10)

X = data.ix[:, [0, 1, 2, 3, 4, 6]].values
X[:, 1] = utils.encode_day_names(X[:, 1])
X[:, 3] = utils.encode_work_flows(X[:, 3])
X[:, 4] = utils.encode_files(X[:, 4])

y = data.ix[:, 5].values


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
plt.savefig('plots/ActualvsFitted.png', format='png')
plt.clf()


#Residual
y_residual = y - y_predicted

fig, ax = plt.subplots()
ax.scatter(y_predicted, y_residual)
ax.set_xlabel('Fitted')
ax.set_ylabel('Residual')
plt.savefig('plots/FittedvsResidual.png', format='png')
plt.clf()

#What do p values mean per column value? -_-
model = sm.ols('SizeofBackupGB ~ WeekNo + DayofWeek + BackupStartTime + WorkFlowID + FileName + BackupTimeHour', data).fit()
print model.summary()
