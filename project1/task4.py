from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import statsmodels.formula.api as sm
from sklearn.model_selection import cross_val_predict, cross_val_score
import utils
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("datasets/housing_data.csv")
data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = data.astype(float)

X = data.ix[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]].values

y = data.ix[:, 13].values

model = sm.ols('MEDV ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT - 1', data).fit()
print model.summary()

###
#Linear Regression Model
###
reg = linear_model.LinearRegression()

y_predicted = cross_val_predict(reg, X, y, cv=10)
print ('RMSE is: ', utils.rmse(y,y_predicted))

cv_scores = cross_val_score(reg, X, y, cv=10, scoring='neg_mean_squared_error')
print ('Average RMSE: ', (sum(cv_scores)/-10.0)**0.5)
print ('Best RMSE: ', np.min((-1 * cv_scores)**0.5))

fig, ax = plt.subplots()
ax.scatter(x=y, y=y_predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()],  'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Fitted')
plt.show()


y_residual = y - y_predicted
fig, ax = plt.subplots()
ax.scatter(y_predicted, y_residual)
ax.set_xlabel('Fitted')
ax.set_ylabel('Residual')
plt.show()


###
#Polynomial Regression Model
###

rmse_degree = []

#Degree beyond 6 takes lot of time to compute
for deg in range(1,6):
    
    regr = make_pipeline(PolynomialFeatures(deg),linear_model.LinearRegression())
    y_predicted = cross_val_predict(regr, X, y, cv = 10)
    cv_scores = cross_val_score(regr, X, y,  cv=10, scoring='neg_mean_squared_error')
    print ('---- Polynomial degree ----', deg)
    print ('RMSE is: ', utils.rmse(y, y_predicted))
    print ('Average RMSE: ', (sum(cv_scores)/-10.0)**0.5)
    print ('Best RMSE: ', np.min((-1 * cv_scores)**0.5))
    rmse_degree.append((sum(cv_scores)/-10.0)**0.5)

    fig, ax = plt.subplots()
    ax.scatter(x=y, y=y_predicted)
    ax.plot([y.min(), y.max()], [y_predicted.min(), y_predicted.max()],  'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Fitted')
    
    plt.show()
    
print ('Average RMSE of all degrees: ', rmse_degree)
plt.figure()
plt.plot(range(1,len(rmse_degree)+1), rmse_degree,color='blue', linewidth=3)
