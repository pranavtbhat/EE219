from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score

from sklearn import linear_model
from sklearn.neural_network import MLPRegressor

import utils

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

data = pd.read_csv("datasets/network_backup_dataset.csv")
kf = KFold(n_splits = 10)

for wid, grp in data.groupby('Work-Flow-ID'):
    X = grp.ix[:, [0, 1, 2, 3, 4, 6]].values
    X[:, 1] = utils.encode_day_names(X[:, 1])
    X[:, 3] = utils.encode_work_flows(X[:, 3])
    X[:, 4] = utils.encode_files(X[:, 4])

    y = grp.ix[:, 5].values

    reg = linear_model.LinearRegression()

    y_predicted = cross_val_predict(reg, X, y, cv=10)

    print('Best RMSE obtained for ', wid, ' was: ', utils.rmse(y,y_predicted))

    fig, ax = plt.subplots()
    ax.scatter(x=y, y=y_predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()],  'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Fitted')
    plt.savefig('plots/ActualvsFitted_' + wid + '.png', format='png')
    plt.show()
    plt.clf()


    # Residual
    y_residual = y - y_predicted

    fig, ax = plt.subplots()
    ax.scatter(y_predicted, y_residual)
    ax.set_xlabel('Fitted')
    ax.set_ylabel('Residual')
    plt.savefig('plots/FittedvsResidual_' + wid + '.png', format='png')
    plt.show()
    plt.clf()
