#!/usr/bin/env python
# -*- coding: utf-8 -*-

from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from utils import get_data, split_data
import scienceplots
plt.style.use("science")


# prepare data
X, y = get_data("../data/MDdata.xlsx", "Sheet1")
X_train, X_test, y_train, y_test = split_data(X, y)

# init models
# params are the output of fine tune results
xgb = XGBRegressor(learning_rate=0.2, max_depth=2, n_estimators=1000)

# training each model
xgb.fit(X_train, y_train)
imps = xgb.feature_importances_ * 100

# figure
plt.figure(figsize=(6, 4), dpi=400)

colors = ['lightblue', 'lightgreen', "orange"]
labels = [r'$\phi$', r"$\gamma$", "T"]
bp = plt.bar([0, 1, 2], imps)
plt.tick_params(axis="x", which="minor", bottom=False, top=False)
plt.xticks([0, 1, 2], labels, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Features", fontsize=15)
plt.ylabel("Importance (\%)", fontsize=15)
plt.xticks(fontsize=15)
plt.ylim(0, 100)

plt.text(-0.08, imps[0] + 3, str(round(imps[0], 1)), fontsize=15)
plt.text(1-0.1, imps[1] + 3, str(round(imps[1], 1)), fontsize=15)
plt.text(2-0.08, imps[2] + 3, str(round(imps[2], 1)), fontsize=15)

plt.savefig("imp.png")
