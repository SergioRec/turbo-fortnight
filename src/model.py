#!/usr/bin/env python
# coding: utf-8
# %%
#importing libraries we need for models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


get_ipython().run_line_magic('matplotlib', 'inline')


# %% [markdown]
# ### Models

# %% [markdown]
# ### Linear Regression

# %%
# Create model object
linear_regressor = LinearRegression()

# Fit model
linear_regressor.fit(X_train, y_train);


# %%
# Generate predictions from data.
y_pred_linear_regressor = linear_regressor.predict(X_test)

# %% [markdown]
# ### Decision Tree Regressor

# %%
# Create model object
decision_tree_regressor = DecisionTreeRegressor()

# Fit model
decision_tree_regressor.fit(X_train, y_train);


# %%
# Generate predictions from data.
y_pred_decision_tree_regressor = decision_tree_regressor.predict(X_test);

# %% [markdown]
# ### Model Evaluation

# %%
