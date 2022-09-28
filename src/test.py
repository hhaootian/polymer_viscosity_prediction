#!/usr/bin/env python
# -*- coding: utf-8 -*-

from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from utils import get_data, split_data


# prepare data
X, y = get_data("../data/MDdata.xlsx", "Sheet1")
X_train, X_test, y_train, y_test = split_data(X, y)

# init models
# params are the output of fine tune results
xgb = XGBRegressor(learning_rate=0.2, max_depth=2, n_estimators=1000)
dt = DecisionTreeRegressor(max_depth=12, random_state=42)
svr = SVR(C=50, degree=1, epsilon=0.1, kernel="rbf")
lr = LinearRegression()

# training each model
xgb.fit(X_train, y_train)
dt.fit(X_train, y_train)
svr.fit(X_train, y_train)
lr.fit(X_train, y_train)

# MSE
print(mean_squared_error(y_test, xgb.predict(X_test)))
print(mean_squared_error(y_test, dt.predict(X_test)))
print(mean_squared_error(y_test, svr.predict(X_test)))
print(mean_squared_error(y_test, lr.predict(X_test)))

# MAPE
print(mean_absolute_percentage_error(y_test, xgb.predict(X_test)))
print(mean_absolute_percentage_error(y_test, dt.predict(X_test)))
print(mean_absolute_percentage_error(y_test, svr.predict(X_test)))
print(mean_absolute_percentage_error(y_test, lr.predict(X_test)))
