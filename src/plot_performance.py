#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_percentage_error, r2_score
)
import matplotlib.pyplot as plt
from utils import get_data, split_data
plt.style.use("science")


# prepare data
X, y = get_data("../data/MDdata.xlsx", "Sheet1")
X_train, X_test, y_train, y_test = split_data(X, y)

# init models
# params are the output of fine tune results
xgb = XGBRegressor(learning_rate=0.2, max_depth=2, n_estimators=1000)
dt = DecisionTreeRegressor(max_depth=12, random_state=42)
svr = SVR(C=50, degree=1, epsilon=0.01, kernel="rbf")
lr = LinearRegression()

# training each model
xgb.fit(X_train, y_train)
dt.fit(X_train, y_train)
svr.fit(X_train, y_train)
lr.fit(X_train, y_train)

# prediction
pred_xgb = xgb.predict(X_test)
pred_dt = dt.predict(X_test)
pred_svr = svr.predict(X_test)
pred_lr = lr.predict(X_test)

# performance
mse = [
    mean_squared_error(pred_xgb, y_test),
    mean_squared_error(pred_dt, y_test),
    mean_squared_error(pred_svr, y_test),
    mean_squared_error(pred_lr, y_test),
]

mape = np.array([
    mean_absolute_percentage_error(pred_xgb, y_test),
    mean_absolute_percentage_error(pred_dt, y_test),
    mean_absolute_percentage_error(pred_svr, y_test),
    mean_absolute_percentage_error(pred_lr, y_test),
]) * 100

r2_xgb = r2_score(y_test, pred_xgb)
r2_dt = r2_score(y_test, pred_dt)
r2_svr = r2_score(y_test, pred_svr)
r2_lr = r2_score(y_test, pred_lr)

# plot MSE and MAPE
plt.figure(figsize=(10, 4), dpi=400)

colors = ['red', 'lightblue', 'lightgreen', "orange", "slategray"]
labels = ["XGBoost", "DT", "SVR", "LR"]

ax = plt.subplot(1, 2, 1)
plt.bar([0, 1, 2, 3], mse)
plt.tick_params(axis="x", which="minor", bottom=False, top=False)
plt.xticks([0, 1, 2, 3], labels, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Models", fontsize=15)
plt.ylabel("Mean Squared Error", fontsize=15)
plt.text(-0.15, 1.05, "(A)", transform=ax.transAxes, fontsize=20)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

ax = plt.subplot(1, 2, 2)
plt.bar([0, 1, 2, 3], mape)
plt.tick_params(axis="x", which="minor", bottom=False, top=False)
plt.xticks([0, 1, 2, 3], labels, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Models", fontsize=15)
plt.ylabel(r"Mean Absolute Percent Error (\%)", fontsize=15)
plt.text(-0.15, 1.05, "(B)", transform=ax.transAxes, fontsize=20)

plt.savefig("../imgs/error.png")

# plot R2
fig = plt.figure(figsize=(12, 10), dpi=400)

ax = fig.add_subplot(2, 2, 1)
ax.scatter(y_test, pred_xgb, facecolors='none', color='red')
plt.xlim(0.4, 4.2)
plt.ylim(0.4, 4.2)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="black", linestyle='--')
plt.xlabel(r"MD calculated $\log(\eta)$", fontsize=15)
plt.ylabel(r"Predicted $\log(\eta)$", fontsize=15)
plt.text(1, 3.5, r"$R^2$ = " + str(round(r2_xgb, 3)), fontsize=20)
plt.title("XGBoost", fontsize=20)
plt.text(-0.15, 1.05, "(A)", transform=ax.transAxes, fontsize=20)

ax = fig.add_subplot(2, 2, 2)
ax.scatter(y_test, pred_dt, facecolors='none', color='red')
plt.xlim(0.4, 4.2)
plt.ylim(0.4, 4.2)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="black", linestyle='--')
plt.xlabel(r"MD calculated $\log(\eta)$", fontsize=15)
plt.ylabel(r"Predicted $\log(\eta)$", fontsize=15)
plt.text(1, 3.5, r"$R^2$ = " + str(round(r2_dt, 3)), fontsize=20)
plt.title("DT", fontsize=20)
plt.text(-0.15, 1.05, "(B)", transform=ax.transAxes, fontsize=20)

ax = fig.add_subplot(2, 2, 3)
ax.scatter(y_test, pred_svr, facecolors='none', color='red')
plt.xlim(0.4, 4.2)
plt.ylim(0.4, 4.2)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="black", linestyle='--')
plt.xlabel(r"MD calculated $\log(\eta)$", fontsize=15)
plt.ylabel(r"Predicted $\log(\eta)$", fontsize=15)
plt.text(1, 3.5, r"$R^2$ = " + str(round(r2_svr, 3)), fontsize=20)
plt.title("SVR", fontsize=20)
plt.text(-0.15, 1.05, "(C)", transform=ax.transAxes, fontsize=20)

ax = fig.add_subplot(2, 2, 4)
ax.scatter(y_test, pred_lr, facecolors='none', color='red')
plt.xlim(0.4, 4.2)
plt.ylim(0.4, 4.2)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="black", linestyle='--')
plt.xlabel(r"MD calculated $\log(\eta)$", fontsize=15)
plt.ylabel(r"Predicted $\log(\eta)$", fontsize=15)
plt.text(1, 3.5, r"$R^2$ = " + str(round(r2_lr, 3)), fontsize=20)
plt.title("LR", fontsize=20)
plt.text(-0.15, 1.05, "(D)", transform=ax.transAxes, fontsize=20)

plt.savefig("../imgs/linear.png")
