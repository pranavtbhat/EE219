from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor

import utils

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

###
# Load the dataset
# Columns:
# 0 -> Week #
# 1 -> Day of Week
# 2 -> Backup Start Time - Hour of Day
# 3 -> Work-Flow-ID
# 4 -> File Name
# 5 -> Size of Backup (GB)
# 6 -> Backup Time (hour)
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
###
reg = linear_model.LinearRegression()

best_coeff = None
best_rmse = float('inf')
best_y = None
best_y_predicted = None

for train_index, test_index in kf.split(X):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    reg.fit(X_train, y_train)
    y_predicted = reg.predict(X_test)

    rmse = utils.rmse(y_predicted, y_test)

    if rmse < best_rmse:
        best_rmse = rmse
        best_coef = reg.coef_
        best_y = y_test
        best_y_predicted = y_predicted

print('Best RMSE obtained was: ', best_rmse)
print('Coefficients were: ', best_coef)

# NO CLUE WHATS GOING ON HERE. ML FOLKS HELP
plt.figure()
plt.scatter(range(len(best_y)), best_y, color='black')
plt.plot(range(len(best_y_predicted)), best_y_predicted, color='blue', linewidth = 2)

plt.show()

###
# Part b: Random Forest Regression
###
rfr = RandomForestRegressor(
        n_estimators = 20,
        max_depth = 4,
        max_features = 6
    )

rfr.fit(X, y)
y_predicted = rfr.predict(X)

rmse = utils.rmse(y, y_predicted)
print('RMSE is ', rmse)


###
# Part c: Neural Networks
###
nnr = MLPRegressor(
        hidden_layer_sizes=(2),
        # Maybe more here?
    )

nnr.fit(X, y)
y_predicted = nnr.predict(X)

rmse = utils.rmse(y, y_predicted)
print('RMSE is ', rmse)
