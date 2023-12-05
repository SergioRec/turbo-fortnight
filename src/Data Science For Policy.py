#!/usr/bin/env python
# coding: utf-8
# %%

# ## Task
# 
# You are asked to carry out an independent analysis on increasing price of energy and its effect on different economic sectors in Great Britain and use this analysis to derive recommendations for informing policy. 
# 
# 
# This can be broken down into three main sections to explore:
# - Rising cost of energy
# - Effects on CPI
# - Effects on a specific sector
# 
# Put forward a clearly defined research question related to this topic and look to carry out appropriate analysis to draw meaningful conclusions. 
# 

# ## Research Question: How the rising cost of energy is effecting the food industry

# **Backgroud**
# 
# Food inflation has been very high in the UK for the past year or so. This has increased the price of weekly shopping for consumers and shops have had to raise the prices of certain items such as milk and eggs. Restaurants have also had to increase the prices of their menu items also.
# 
# Currently food inflation is still increasing but the rate at which it is doing so is slowing down compred to previous months. This is due to a few factors including the war in Ukraine, supply chain issues and rising energy prices. 

# ### Import required libraries

# %%


#importing libraries we need
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

get_ipython().run_line_magic('matplotlib', 'inline')


# %%


# Load in dataset
increase_in_price = pd.read_csv("C:/Users/onalaj/New Documents/Admin & Learning/Data Science Campus Modules/For Policy/increase_in_the_price_level_since_the_pandemic.csv") 


# %%


# Check dataframe loaded in correctly
increase_in_price.head()


# %%


# Change column name for date
increase_in_price.columns = ["Date", "Canada", "France", "Germany", "Italy", "Japan", "UK", "US"]

# Filter so dates only January
increase_in_price = increase_in_price.loc[increase_in_price['Date'].str.startswith('Jan')]


# %%


# Create a line plot
plt.figure(figsize=(15,6))
plt.xlabel('Date')
plt.ylabel('Consumer Price Index')
plt.title('Figure 1: Increase in the price level since the pandemic in the UK')
plt.plot(increase_in_price["Date"].values, increase_in_price["UK"].values, color='red')

plt.show();


# Clear increase in CPI since January 2021

# ### Food inflation

# %%


# Load in dataset
cpi_food_inflation = pd.read_csv("C:/Users/onalaj/New Documents/Admin & Learning/Data Science Campus Modules/For Policy/cpi_food_inflation.csv") 


# %%


# Change column name for date
cpi_food_inflation.columns = ["Date", "Canada", "France", "Germany", "Italy", "Japan", "UK", "US"]

# Filter so dates only January
cpi_food_inflation = cpi_food_inflation.loc[cpi_food_inflation['Date'].str.startswith('Jan')]


# %%


# Create a line plot
plt.figure(figsize=(15,6))
plt.xlabel('Date')
plt.ylabel('Food CPI inflation')
plt.title('Figure 2: Rate of food CPI inflation in the UK')
plt.plot(cpi_food_inflation["Date"].values, cpi_food_inflation["UK"].values, color='blue')

plt.show();


# Large increase in the rate of food CPI inflation since January 2021 not really linked to covid

# ### Global food commodities

# %%


# Load in dataset
global_food_commodities = pd.read_csv("C:/Users/onalaj/New Documents/Admin & Learning/Data Science Campus Modules/For Policy/prices_of_global_food_commodities.csv") 


# %%


# Change column name for date
global_food_commodities.columns = ["Date", "Food Price Index", "Meat", "Dairy", "Cereals", "Oils", "Sugar"]

# Filter so dates only January
global_food_commodities = global_food_commodities.loc[global_food_commodities['Date'].str.startswith('Jan')]


# %%


# Create a line plot
plt.figure(figsize=(15,6))
plt.xlabel('Date')
plt.ylabel('Index')
plt.title('Figure 3: Prices of global food commodities')

plt.plot(global_food_commodities["Date"].values, global_food_commodities["Food Price Index"].values, color='red', label = "Food Price Index")
plt.plot(global_food_commodities["Date"].values, global_food_commodities["Meat"].values, color='orange', label = "Meat")
plt.plot(global_food_commodities["Date"].values, global_food_commodities["Dairy"].values, color='yellow', label = "Dairy")
plt.plot(global_food_commodities["Date"].values, global_food_commodities["Cereals"].values, color='green', label = "Cereals")
plt.plot(global_food_commodities["Date"].values, global_food_commodities["Oils"].values, color='blue', label = "Oils")
plt.plot(global_food_commodities["Date"].values, global_food_commodities["Sugar"].values, color='indigo', label = "Sugar")

plt.legend()
plt.show();


# Clear to see a downward trend in prices of global food commodities from mid-2022 so surprising to see this hasn't been reflected in UK food inflation as of yet. 

# ### Energy price inflation

# %%


# Load dataset
energy_price_inflation = pd.read_csv("C:/Users/onalaj/New Documents/Admin & Learning/Data Science Campus Modules/For Policy/energy_price_inflation.csv") 


# %%


# Change column name for date
energy_price_inflation.columns = ["Date", "Canada", "France", "Germany", "Italy", "Japan", "UK", "US"]

# Filter so dates only January
energy_price_inflation = energy_price_inflation.loc[energy_price_inflation['Date'].str.startswith('Jan')]


# %%


# Create a line plot
plt.figure(figsize=(15,6))
plt.xlabel('Date')
plt.ylabel('Rate of energy CPI inflation (%)')
plt.title('Figure 4: Energy price inflation')

plt.plot(energy_price_inflation["Date"].values, energy_price_inflation["Canada"].values, color='indigo', label = "Canada")
plt.plot(energy_price_inflation["Date"].values, energy_price_inflation["France"].values, color='orange', label = "France")
plt.plot(energy_price_inflation["Date"].values, energy_price_inflation["Germany"].values, color='yellow', label = "Germany")
plt.plot(energy_price_inflation["Date"].values, energy_price_inflation["Italy"].values, color='green', label = "Italy")
plt.plot(energy_price_inflation["Date"].values, energy_price_inflation["Japan"].values, color='blue', label = "Japan")
plt.plot(energy_price_inflation["Date"].values, energy_price_inflation["UK"].values, color='red', label = "UK")
plt.plot(energy_price_inflation["Date"].values, energy_price_inflation["US"].values, color='violet', label = "US")


plt.legend()
plt.show();


# Compared to other G7 countries the UK's enery inflation is much higher currently and in January 2022 didn't start to drop like the other countries. Only in Italy is the situation similar.

# **Energy price cap**
# 
# The energy price gap which is calculated every 3 months is regulated by Ofgem. They decide what it should be and it is dictated by wholesale gas prices which is why it takes 3 months to come into affect. Wholesale gas prices are extremely volatile and thus difficult to predict
# 
# Machine learning models will now be used to try and predict what the price cap will be 
# 

# ### Pipeline start

# %%


price_cap = pd.read_csv("C:/Users/onalaj/New Documents/Admin & Learning/Data Science Campus Modules/For Policy/retail-price-comparison.csv") 


# %%


price_cap


# %%


price_cap.columns = ["Date", 
                     "Average standard variable tariff - Legacy Supplier", 
                     "Average standard variable tariff - Other Suppliers", 
                     "Average fixed tariff", "Cheapest tariff - Legacy Supplier", 
                     "Cheapest tariff - All Suppliers", 
                     "Cheapest tariff (Basket)", 
                     "Price Cap"]


# %%


# get information about the data in the frame.
price_cap.info()


# Null values in `Average fixed tariff` and `Price Cap` columns 

# ### Data cleaning

# %%


# convert date to datetime
price_cap['Date'] = price_cap['Date'].astype('datetime64[ns]')


# %%


# check variable distribution
price_cap['Average fixed tariff'].plot(kind = 'hist')


# %%


# check variable distribution
price_cap['Price Cap'].plot(kind = 'hist')


# Will fill missing values with the median as variables aren't distributed in a symetric manner

# #### Replace with median

# %%


# Calculate the median of Average fixed tariff.
average_fixed_tariff_median = price_cap['Average fixed tariff'].median()

# Calculate the median of Price Cap.
price_cap_median = price_cap['Price Cap'].median()


# %%


# Fill in the missing values with the median value.
price_cap['Average fixed tariff'] = price_cap['Average fixed tariff'].fillna(average_fixed_tariff_median)

price_cap['Price Cap'] = price_cap['Price Cap'].fillna(price_cap_median)


# %%


price_cap['Average fixed tariff']


# %%


price_cap['Price Cap']


# %%


# check no missing values
price_cap.isna().sum()


# #### Correlation

# %%


# Select only the numerical data.
numerical_data = price_cap.select_dtypes(include='number')

# Generate the correlation matrix with pandas.
correlation_matrix = numerical_data.corr()

correlation_matrix.round(2)


# Looking at our table, the variables `Average standard variable tariff - Legacy Supplier`, `Average standard variable tariff - Other Suppliers`	`Cheapest tariff - Legacy Supplier`, `Cheapest tariff - All Suppliers`, `Cheapest tariff (Basket)` are correlated to a significant degree with the price cap variable. `Average fixed tariff`is less so.
# 
# 

# #### Feature selection

# %%


# Remove the date feature from the dataframe.
price_cap = price_cap.drop(columns=['Date'])

price_cap.head()


# #### Automatic feature selection

# - The target data is a continuous variable, so regression will be used to predict its value. 
# - To select the most suitable features a regression based statistical test should be used
# - The f_regresssion measure will be used

# %%


# Get the target data column into it's own series object.
y = price_cap['Price Cap']

# Remove target
X = price_cap.drop(columns=['Price Cap'])
X.head()


# %%


# 6 features in the data set, 3 wanted.
number_of_desired_features = 3

# Create the selector object
fr_selector = SelectKBest(score_func = f_regression, k = number_of_desired_features)

fr_selector.fit(X = X, y = y)

# We then get an object that has True/False values of whether the number_of_desired_features are in the best set.
columns_selected = fr_selector.get_support()

# Get the columns from our frame which correspond to the True/False values.
selected_cols = X.columns[columns_selected].to_list()
print("Top {} Columns are:\n\n{}".format(number_of_desired_features, selected_cols))

# Get the frame with only the columns we want.
selected_data = X[selected_cols]

selected_data.head()


# ### Machine Learning model - Regression

# %%


# Get the target data column into it's own series object.
y = price_cap['Price Cap'].to_numpy()

# Remove target and uneeded features
X = price_cap.drop(columns=['Price Cap', "Average fixed tariff", 
                            "Cheapest tariff - Legacy Supplier", 
                            "Cheapest tariff - All Suppliers"])

# feature column names
feature_columns = X.columns

# Create scaler object
rb_scaler = RobustScaler()

# Scale the features
X = rb_scaler.fit_transform(X)
X = pd.DataFrame(X, columns = feature_columns)
X.head()


# ##### Model training - Linear Regression

# %%


# Using all of the columns from the scaled data.
X = X.to_numpy()

# Split the training and test data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Create model object
linear_model_multi_variable = LinearRegression()

# Fit model
linear_model_multi_variable.fit(X_train, y_train);


# %%


# Generate predictions from data.
y_pred_linear_multi = linear_model_multi_variable.predict(X_test)


# ##### Model training - Decision Tree Regressor

# %%


# Create model object
decision_tree_regressor = DecisionTreeRegressor()

# Fit model
decision_tree_regressor.fit(X_train, y_train);


# %%


# Generate predictions from data.
y_pred_dtree_regress = decision_tree_regressor.predict(X_test)


# ##### Model training - Support Vector Regression 

# %%


# Create a support vector regression model
support_vector_regression = SVR(kernel='linear')
 
# Fit model
support_vector_regression.fit(X_train, y_train)


# %%


# Predict the response for a new data point
y_pred_svr = support_vector_regression.predict(X_test)


# ### Evaluating models
# 
# In order to see whether the used model is good at making predictions it needs to be evaluated. There are a few methods of doing this.

# $R^2$ measure is a commonly used metric to see how well a regression performs. It quantifies the proportion of variance that our training data predicts in the target data.
# 
# Bad models will produce a value of  $R^2$ ∼ 0 and very bad models will be  $R^2$ < 0

# ### $R^2$ evaluation

# ##### Linear Regression

# %%


# Calculate R^2 value using the true and predicted values of y_test.
r2_value_linear_multi = r2_score(y_test, y_pred_linear_multi)

print("Multivariable R^2 value: \n", r2_value_linear_multi)


# test size of 0.2 and random state 1234
# 
# Multivariable $R^2$ value: 0.9432097484784084
# 
# <br>
# 
# test size of 0.3 and random state 1234
# 
# Multivariable $R^2$ value: 0.9626036002540803
# 
# <br>
# 
# test size of 0.3 and random state 100
# 
# Multivariable $R^2$ value: 0.9735712614554296

# ##### Decision Tree Regressor

# %%


# Calculate R^2 value using the true and predicted values of y_test.
r2_value_dtree_regress = r2_score(y_test, y_pred_dtree_regress)

print("Multivariable R^2 value: \n", r2_value_dtree_regress)


# test size of 0.3 and random state 100
# 
# Multivariable  $𝑅^2$ value: 0.9926610817287851
# 
# <br>
# 
# test size of 0.2 and random state 100
# 
# Multivariable $R^2$ value: 0.996908585279634

# ###### Support Vector Regression 

# %%


# Calculate R^2 value using the true and predicted values of y_test.
r2_value_svr = r2_score(y_test, y_pred_svr)

print("Multivariable R^2 value: \n", r2_value_svr)


# test size of 0.3 and random state 100
# 
# Multivariable $R^2$ value: 0.9517165024961327
# 
# <br>
# 
# test size of 0.2 and random state 100
# 
# Multivariable $R^2$ value: 0.9637441879569616

# ### Cross validation evaluation

# ###### Linear Regression

# %%


# Set the chosen K
K = 5

# Create model object
linear_model = LinearRegression()

# General cross-validation scores using the model pre-split data.
cv_scores = cross_val_score(estimator=linear_model, X=X, y=y, cv=K, scoring="r2")

print(cv_scores.mean())


# %%


# Predict the test values using cross validation.
y_pred = cross_val_predict(estimator=linear_model, X=X, y=y, cv=K)

# We can then compare our predicted and true values using a plot.
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_pred, edgecolors=(0, 0, 0), alpha=0.4)
ax.plot([0, y.max()], [0, y.max()], 'k--', lw=4)
ax.set_xlabel('Measured Price Cap')
ax.set_ylabel('Predicted Price Cap')
ax.set_title(r"True and Predicted $y_i$")
plt.show()


# #### Decision tree regressor

# %%


# Set the chosen K
K = 5

# Create model object
dtree_regressor = DecisionTreeRegressor()

# General cross-validation scores using the model pre-split data.
cv_scores = cross_val_score(estimator=dtree_regressor, X=X, y=y, cv=K, scoring="r2")

print(cv_scores.mean())


# %%


# Predict the test values using cross validation.
y_pred_dtree = cross_val_predict(estimator=dtree_regressor, X=X, y=y, cv=K)

# We can then compare our predicted and true values using a plot.
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_pred_dtree, edgecolors=(0, 0, 0), alpha=0.4)
ax.plot([0, y.max()], [0, y.max()], 'k--', lw=4)
ax.set_xlabel('Measured Price Cap')
ax.set_ylabel('Predicted Price Cap')
ax.set_title(r"True and Predicted $y_i$")
plt.show()


# #### SVR

# %%


# Set the chosen K
K = 5

# Create model object
svr = SVR(kernel='linear')

# General cross-validation scores using the model pre-split data.
cv_scores = cross_val_score(estimator=svr, X=X, y=y, cv=K, scoring="r2")

print(cv_scores.mean())


# %%


# Predict the test values using cross validation.
y_pred_support_vr = cross_val_predict(estimator=svr, X=X, y=y, cv=K)

# We can then compare our predicted and true values using a plot.
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_pred_support_vr, edgecolors=(0, 0, 0), alpha=0.4)
ax.plot([0, y.max()], [0, y.max()], 'k--', lw=4)
ax.set_xlabel('Measured Price Cap')
ax.set_ylabel('Predicted Price Cap')
ax.set_title(r"True and Predicted $y_i$")
plt.show()


# ### Mean squared error

# #### Linear Regression

# %%


# Create Linear Regression Object #
linear_model = LinearRegression()

# Fit 
linear_model.fit(X_train, y_train)

# Obtain predictions #
y_pred_lr_mse = linear_model.predict(X_test)

# Obtain the MSE #
MSE = mean_squared_error(y_test, y_pred_lr_mse)

# Obtain the RMSE #
RMSE = mean_squared_error(y_test, y_pred_lr_mse, squared = False)

print('The MSE of the multivariate model is: \n {}'.format(MSE))
print('The RMSE of the multivariate model is: \n {}'.format(RMSE))


# #### Decision tree regressor 

# %%


# Create Linear Regression Object #
dtree_regressor = DecisionTreeRegressor()

# Fit 
dtree_regressor.fit(X_train, y_train)

# Obtain predictions #
y_pred_dtree_mse = dtree_regressor.predict(X_test)

# Obtain the MSE #
MSE = mean_squared_error(y_test, y_pred_dtree_mse)

# Obtain the RMSE #
RMSE = mean_squared_error(y_test, y_pred_dtree_mse, squared = False)

print('The MSE of the decision tree regressor model is: \n {}'.format(MSE))
print('The RMSE of the decision tree regressor is: \n {}'.format(RMSE))


# #### SVR

# %%


# Create Linear Regression Object #
svr = SVR(kernel='linear')

# Fit 
svr.fit(X_train, y_train)

# Obtain predictions #
y_pred_svr_mse = svr.predict(X_test)

# Obtain the MSE #
MSE = mean_squared_error(y_test, y_pred_svr_mse)

# Obtain the RMSE #
RMSE = mean_squared_error(y_test, y_pred_svr_mse, squared = False)

print('The MSE of the svr model is: \n {}'.format(MSE))
print('The RMSE of the svr model is: \n {}'.format(RMSE))


# ### Model summaries
# 
# - All models had high $R^2$ values but the `Decision Tree Regressor` performed best in this case
# 
# 
# - The `Decision Tree Regressor`had the lowest MSE and RMSE
# 
# 
# - The `Decision Tree Regressor`appears to be the best model for predicting the price cap by making use of `Average standard variable tariff - Legacy Supplier`, `Average standard variable tariff - Other Suppliers` and `Cheapest tariff (Basket)` features

# ### Recommendations for informing policy
# 
# - Can inform businesses in the food industry to maybe get a fixed energy deal at a certain time if the predictions are correct

# #### Areas for improvement
# 
# [ ] Better dataset with more data as probably 130 is not enough for a machine learning model
# 
# [ ] Try the model with different features

# %%




