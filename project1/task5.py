from sklearn.linear_model import RidgeCV, LassoCV
import utils
import pandas as pd

data = pd.read_csv("datasets/housing_data.csv")

X = data.ix[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]].values

Y = data.ix[:, 13].values

# Ridge regression

tuningAlpha = [1,0.1,0.01,0.001]

ridge = RidgeCV(normalize=True,alphas=tuningAlpha, cv=10)
ridge.fit(X, Y)
prediction = ridge.predict(X)

print "Best Alpha value for Ridge Regression : " + str(ridge.alpha_)
print 'Best RMSE for corresponding Alpha =', utils.rmse(Y, prediction)


# Lasso Regression

tuningAlpha = [1,0.1,0.01,0.001]
lasso = LassoCV(normalize=True, alphas=tuningAlpha, cv=10)
lasso.fit(X,Y)
prediction = lasso.predict(X)

print "Best Alpha value for Lasso Regularization : " + str(lasso.alpha_)
print 'Best RMSE for corresponding Alpha =', utils.rmse(Y, prediction)
