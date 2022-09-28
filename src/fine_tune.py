#!/usr/bin/env python
# -*- coding: utf-8 -*-

from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from utils import get_data, split_data, grid_search


# prepare data
X, y = get_data("../data/MDdata.xlsx", "Sheet1")
X_train, X_test, y_train, y_test = split_data(X, y)

# init models
xgb = XGBRegressor()
dt = DecisionTreeRegressor(random_state=42)
svr = SVR()
lr = LinearRegression()

# CV for XGBRegressor
xgb_param_grid = {
    "learning_rate": [0.05, 0.1, 0.2, 0.3],
    "n_estimators": [500, 1000, 1500],
    "max_depth": [1, 2, 3]
}
xgb_best_params, xgb_best_score = grid_search(
    X_train, y_train, xgb, xgb_param_grid
)
print(xgb_best_params)

# CV for DecisionTreeRegressor
dt_param_grid = {
    "max_depth": list(range(2, 15, 2))
}
dt_best_params, dt_best_score = grid_search(
    X_train, y_train, dt, dt_param_grid
)
print(dt_best_params)

# CV for SVR
svr_param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [1, 2, 3],
    'C': [1, 10, 50],
    'epsilon': [0.01, 0.1, 0.2]
}
svr_best_params, svr_best_score = grid_search(
    X_train, y_train, svr, svr_param_grid
)
print(svr_best_params)
