from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
import statsmodels.formula.api as sm
from scipy.stats import randint as sp_randint

from sklearn.grid_search import RandomizedSearchCV,GridSearchCV

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


X = data.ix[:, [ 0,1, 2, 3, 4, 6]].values
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

dataNew=data
dataNew['WorkFlowID']=networkDataset['WorkFlowID']
dataNew['FileName']=networkDataset['FileName']
del dataNew['SizeofBackupGB']
# del dataNew['FileName']
# del dataNew['WeekNo']
# del dataNew['WorkFlowID']


###
# Part b: Random Forest Regression
###
#
# rmse=[]
# trees=[]
# for noOfTrees in range(10,400,10):
#     rfr = RandomForestRegressor(
#         n_estimators=noOfTrees,
#         max_depth=10,
#         max_features=4
#     )
#
#     cv_scores = cross_val_score(rfr, X, y, cv=10, scoring='neg_mean_squared_error')
#     print noOfTrees
#
#     rmse.append((sum(cv_scores) / -10.0) ** 0.5)
#     trees.append(noOfTrees)
#
# plt.plot(trees,rmse)
# plt.xlabel('Number of Trees')
# plt.ylabel('RMSE')
# plt.savefig('plots/rrRmseVsTrees-maxDepth10-6features2.png',format='png')
# plt.clf()

# rmse=[]
# depths=[]
# for depth in range(4,15,1):
#     rfr = RandomForestRegressor(
#         n_estimators=80,
#         max_depth=depth,
#         max_features=4
#     )
#
#     cv_scores = cross_val_score(rfr, X, y, cv=10, scoring='neg_mean_squared_error')
#     print depth
#
#     rmse.append((sum(cv_scores) / -10.0) ** 0.5)
#     depths.append(depth)
#
# plt.plot(depths,rmse)
# plt.xlabel('Max Depth')
# plt.ylabel('RMSE')
# plt.savefig('plots/rrRmseVsDepths-nestimators80-4features2.png',format='png')


rfr = RandomForestRegressor(
        n_estimators=160,
        max_depth=10,
        max_features=4
    )
y_predicted = cross_val_predict(rfr, dataNew, y, cv=10)

cv_scores = cross_val_score(rfr, dataNew, y, cv=10, scoring='neg_mean_squared_error')
print ((sum(cv_scores) / -10.0) ** 0.5)

rfr.fit(dataNew,y)
print rfr.feature_importances_
print dataNew

print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), rfr.feature_importances_), dataNew.columns),
             reverse=True)

fig, ax = plt.subplots()
ax.scatter(x=y, y=y_predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()],  'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Fitted')
plt.savefig('plots/rrActualvsFitted-nestimators180-maxDepth10-6features2.png', format='png')
plt.clf()


#Residual
y_residual = y - y_predicted

fig, ax = plt.subplots()
ax.scatter(y_predicted, y_residual)
ax.set_xlabel('Fitted')
ax.set_ylabel('Residual')
plt.savefig('plots/rrFittedvsResidual-nestimators180-maxDepth10-6features2.png', format='png')
plt.clf()

# clf = RandomForestRegressor()
# param_dist = {"n_estimators":sp_randint(1, 180),
#               "max_depth": sp_randint(4, 14),
#               "max_features": sp_randint(1, 6)}
#
# #Tune Random Forest using Randomized Search
# n_iter_search = 20
# random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                    n_iter=n_iter_search)
# random_search.fit(X, y)
# print('Best Parameters for Random forest:')
# print(random_search.best_params_)