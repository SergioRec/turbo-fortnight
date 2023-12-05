# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# Step 1

# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# %%
ames_data = pd.read_csv("../data/ames.csv")

# %%
zones = ["RL", "RM", "RH"]
ames_data = ames_data[ames_data["MS Zoning"].isin(zones)]

# %%
ames_data["Total Bathrooms"] = ames_data["Bsmt Full Bath"] + ames_data["Bsmt Half Bath"] + ames_data["Full Bath"] + ames_data["Half Bath"]

# %%
ames_data = ames_data[["Lot Area", "Overall Qual", "Overall Cond", "Gr Liv Area", "TotRms AbvGrd", "Total Bathrooms", "Garage Area", "Bldg Type", "SalePrice"]]
ames_data = ames_data.dropna()
ames_data["Lot Area"] = np.log(ames_data["Lot Area"])
ames_data["Gr Liv Area"] = np.log(ames_data["Gr Liv Area"])
ames_data["SalePrice"] = np.log(ames_data["SalePrice"])

# %%
ames_data = ames_data.astype({"Total Bathrooms":"int64"})

# %% [markdown]
# ### Linear Regression

# %%
# Create model object
linear_regressor = LinearRegression()

# Fit model
linear_regressor.fit(X_train, y_train);

# Generate predictions from data.
y_pred_linear_regressor = linear_regressor.predict(X_test)

# %% [markdown]
# ### Decision Tree Regressor

# %%
# Create model object
decision_tree_regressor = DecisionTreeRegressor()

# Fit model
decision_tree_regressor.fit(X_train, y_train);

# Generate predictions from data.
y_pred_decision_tree_regressor = decision_tree_regressor.predict(X_test);

# %% [markdown]
# ### Model Evaluation

# %%
