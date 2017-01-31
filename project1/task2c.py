from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
import statsmodels.formula.api as sm

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import utils
from sklearn import cross_validation
import numpy as np


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
# Part c: Neural Networks
###
# for i in range(1,100,1):
#     nnr = MLPRegressor(hidden_layer_sizes=(i))# Maybe more here?)
#
#     nnr.fit(X, y)
#     y_predicted = nnr.predict(X)
#
#     rmse = utils.rmse(y, y_predicted)
#     print('RMSE is ', rmse)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=3)

#Neural Networks
ds = SupervisedDataSet(6, 1)
ds.setField( 'input', X_train )
y_train_nn = y_train.copy().reshape(-1, 1)
ds.setField( 'target', y_train_nn )

ds_test = SupervisedDataSet(6, 1)
ds_test.setField( 'input', X_test)
y_test_nn = y_test.copy().reshape( -1, 1 )
ds_test.setField( 'target', y_test_nn )

for hidden in range(5,6):
    hidden_size = hidden

    net = buildNetwork(6, hidden_size, 1, bias = True)

    trainer = BackpropTrainer( net, ds )

    trnerror, valerror = trainer.trainUntilConvergence(maxEpochs = 200)
    plt.plot(trnerror,'b',valerror,'r')
    plt.savefig("plots/nn"+str(hidden)+"_100"+".png",format='png')
    plt.clf()

    p = net.activateOnDataset( ds_test )
    print('Neural Network - Hidden size: %d Epchs: %d RMSE: %.4f' % (hidden_size, 100, np.sqrt(np.sum((p - y_test_nn) ** 2)/y_test.size)))


def printNetwork():
    print(net)
    print(net.modules)
    for mod in net.modules:
        print("Module:", mod.name)
        if mod.paramdim > 0:
            print("--parameters:", mod.params)
        for conn in net.connections[mod]:
            print("-connection to", conn.outmod.name)
            if conn.paramdim > 0:
                print("- parameters", conn.params)
        if hasattr(net, "recurrentConns"):
            print("Recurrent connections")
            for conn in net.recurrentConns:
                print("-", conn.inmod.name, " to", conn.outmod.name)
                if conn.paramdim > 0:
                    print("- parameters", conn.params)
