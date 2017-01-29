from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import utils

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("datasets/housing_data.csv")
splits =  10
kf = KFold(n_splits = splits)

X = data.ix[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]].values

y = data.ix[:, 13].values


###
#Linear Regression Model
###
reg = linear_model.LinearRegression()

best_rmse = float('inf')
best_y = None
best_y_predicted = None
best_poly_y_predicted = None
residual = None
rmse_sum = 0

for train_index, test_index in kf.split(X):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    reg.fit(X_train, y_train)
    y_predicted = reg.predict(X_test)
    rmse = utils.rmse(y_predicted, y_test)
    rmse_sum = rmse_sum + rmse
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_coef = reg.coef_
        best_y = y_test
        best_y_predicted = y_predicted
        residual = y_predicted - y_test
        
print('Best RMSE obtained from linear regression was: ', best_rmse)
print('Average RMSE obtained from linear regression was: ', rmse_sum/splits)

plt.figure()
plt.scatter(range(len(best_y)), best_y, color='black')
plt.plot(range(len(best_y_predicted)), best_y_predicted, color='blue', linewidth = 2)

plt.show()

plt.figure()
plt.scatter(range(len(residual)), residual, color='black')
plt.plot(range(len(best_y_predicted)), best_y_predicted, color='blue', linewidth = 2)

plt.show()     


###
#Polynomial Regression Model
###

rmse_sum = 0
best_rmse = float('inf')
rmse_degree = []

for deg in range(1,4):
    poly = PolynomialFeatures(degree=deg)
    rmse_sum = 0
    best_rmse = float('inf')
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]    
        Xtrain_ = poly.fit_transform(X_train)
        Xtest_ = poly.fit_transform(X_test)
    
        reg.fit(Xtrain_, y_train)
        y_poly_predicted = reg.predict(Xtest_)    
        rmse = utils.rmse(y_poly_predicted, y_test)
        rmse_sum = rmse_sum + rmse
    
        if rmse < best_rmse:
            best_rmse = rmse
            best_y = y_test
            best_poly_y_predicted = y_poly_predicted
            residual = y_poly_predicted - y_test
    
    
    print('Best RMSE obtained from polynomial regression with degree:', deg,  'was: ', best_rmse)
    print('Average RMSE obtained from polynomial regression with degree: ', deg, 'was: ', rmse_sum/splits)

    plt.figure()
    plt.scatter(range(len(best_y)), best_y, color='black')
    plt.plot(range(len(best_poly_y_predicted)), best_poly_y_predicted, color='blue', linewidth = 2)
    
    plt.show()
    
    plt.figure()
    plt.scatter(range(len(residual)), residual, color='black')
    plt.plot(range(len(best_poly_y_predicted)), best_poly_y_predicted, color='blue', linewidth = 2)
    
    plt.show()     
    rmse_degree.append(best_rmse)


print('All rmse are: ', rmse_degree)
plt.figure()
plt.plot(range(len(rmse_degree)), rmse_degree,color='blue', linewidth=3)