# %%
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %%
# Step 1
ames_data = pd.read_csv("../data/ames.csv")

# %%
zones = ["RL", "RM", "RH"]
ames_data = ames_data[ames_data["MS Zoning"].isin(zones)]
ames_data["Total Bathrooms"] = (
    ames_data["Bsmt Full Bath"]
    + ames_data["Bsmt Half Bath"]
    + ames_data["Full Bath"]
    + ames_data["Half Bath"]
)
ames_data = ames_data[
    ["Lot Area", "Overall Qual", "Overall Cond", "Gr Liv Area",
     "TotRms AbvGrd", "Total Bathrooms", "Garage Area", "Bldg Type",
     "Total Bsmt SF", "SalePrice"]
]

ames_data = ames_data.dropna()
ames_data["Lot Area"] = np.log(ames_data["Lot Area"])
ames_data["Gr Liv Area"] = np.log(ames_data["Gr Liv Area"])
ames_data["SalePrice"] = np.log(ames_data["SalePrice"])
ames_data = ames_data.astype({"Total Bathrooms": "int64"})

# %%
# Step 2

# %%
# list of columns that need scaling
cols = ["Overall Qual", "Overall Cond", "Lot Area", "Gr Liv Area",
        "TotRms AbvGrd", "Total Bsmt SF", "Garage Area", "Total Bathrooms"]

# subset dataset to include target columns + outcome variable
df_sub = ames_data[cols + ["SalePrice"] + ["Bldg Type"]].copy()

# splits between train and test datasets
train, test = train_test_split(df_sub, test_size=0.2)


# %%
#scaling training and test data function
def scale_data(train_df, test_df, columns):
    """Scale train and test data"""
    scaler = MinMaxScaler()
    train_df[columns] = scaler.fit_transform(train_df[columns])
    test_df[columns] = scaler.transform(test_df[columns])

    return train_df, test_df


# %%
train_scaled, test_scaled = scale_data(train, test, cols)

# One Hot Encoding
one_hot_encoder = OneHotEncoder()

def one_hot_encoder_bldg_type(df_scaled):
    """One hot encoder for train and test scaled data"""
    data_to_encode = (
        pd.DataFrame(data=df_scaled['Bldg Type'], columns=['Bldg Type'])
    )
    new_data_array = (
        one_hot_encoder.fit_transform(data_to_encode[['Bldg Type']]).toarray()
    )
    new_column_names = one_hot_encoder.get_feature_names_out(['Bldg Type'])
    new_data = pd.DataFrame(data=new_data_array, columns=new_column_names)
    ames_data_encoded = (
        pd.concat(
            [
                new_data.reset_index(drop=True),
                df_scaled.reset_index(drop=True)
            ],
            axis=1
        )
    )
    ames_data_encoded = ames_data_encoded.drop('Bldg Type', axis=1)
    return ames_data_encoded


ames_data_encoded_train_scaled = one_hot_encoder_bldg_type(train_scaled)
ames_data_encoded_test_scaled = one_hot_encoder_bldg_type(test_scaled)

ames_data_encoded_train = one_hot_encoder_bldg_type(train)
ames_data_encoded_test = one_hot_encoder_bldg_type(test)

# %% [markdown]
# ### Linear Regression

# For the training set(scaled)
X_train_scaled = ames_data_encoded_train_scaled.drop('SalePrice', axis=1)  # Features: Drop the target variable
y_train_scaled = ames_data_encoded_train_scaled['SalePrice']  # Target: Only the 'SalePrice' column

# For the testing set(scaled)
X_test_scaled = ames_data_encoded_test_scaled.drop('SalePrice', axis=1)  # Features: Drop the target variable
y_test_scaled = ames_data_encoded_test_scaled['SalePrice']  # Target: Only the 'SalePrice' column

# For the training set(non-scaled)
X_train = ames_data_encoded_train.drop('SalePrice', axis=1)  # Features: Drop the target variable
y_train = ames_data_encoded_train['SalePrice']  # Target: Only the 'SalePrice' column

# For the testing set(non-scaled)
X_test = ames_data_encoded_test.drop('SalePrice', axis=1)  # Features: Drop the target variable
y_test = ames_data_encoded_test['SalePrice']  # Target: Only the 'SalePrice' column

# %%
# Create model object
model_LR = LinearRegression()

# Fit model
model_LR.fit(X_train_scaled, y_train_scaled)


# %% [markdown]
# ### Model Evaluation

# %%
# Cross-validation for RMSE
rmse_scores = np.sqrt(
    -cross_val_score(
        model_LR,
        X_train_scaled,
        y_train,
        cv=5,
        scoring='neg_mean_squared_error'
    )
)

# Cross-validation for MSE
mse_scores = -cross_val_score(
    model_LR,
    X_train_scaled,
    y_train_scaled,
    cv=5,
    scoring='neg_mean_squared_error'
)

# Train the model
model_LR.fit(X_train_scaled, y_train_scaled)

# Make predictions using the scaled test set
y_pred = model_LR.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test_scaled, y_pred)
mse = mean_squared_error(y_test_scaled, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_scaled, y_pred)

# Print cross-validation scores and evaluation metrics
print("Linear Regression:")
print("Cross-Validation RMSE scores:", rmse_scores)
print("Mean of Cross-Validation RMSE scores:", rmse_scores.mean())
print("Standard Deviation of Cross-Validation RMSE scores:", rmse_scores.std())
print("Cross-Validation MSE scores:", mse_scores)
print("Mean of Cross-Validation MSE scores:", mse_scores.mean())
print("Standard Deviation of Cross-Validation MSE scores:", mse_scores.std())
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared: {r2}\n")


# %% [markdown]
# ### Decision Tree Regressor

# %%
# Create model object
model_DT = DecisionTreeRegressor()

# Fit model
model_DT.fit(X_train, y_train)

# %% [markdown]
# ### Model Evaluation

# %%
# Cross-validation for RMSE
rmse_scores = np.sqrt(
    -cross_val_score(
        model_DT,
        X_train,
        y_train,
        cv=5,
        scoring='neg_mean_squared_error'
    )
)

# Cross-validation for MSE
mse_scores = -cross_val_score(
    model_DT,
    X_train,
    y_train,
    cv=5,
    scoring='neg_mean_squared_error'
)

# Train the model
model_DT.fit(X_train, y_train)

# Make predictions using the scaled test set
y_pred = model_DT.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print cross-validation scores and evaluation metrics
print("Decision Tree:")
print("Cross-Validation RMSE scores:", rmse_scores)
print("Mean of Cross-Validation RMSE scores:", rmse_scores.mean())
print("Standard Deviation of Cross-Validation RMSE scores:", rmse_scores.std())
print("Cross-Validation MSE scores:", mse_scores)
print("Mean of Cross-Validation MSE scores:", mse_scores.mean())
print("Standard Deviation of Cross-Validation MSE scores:", mse_scores.std())
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared: {r2}\n")

# Initialize the Random Forest Regressor
model_RF = RandomForestRegressor()

# Cross-validation for RMSE
rmse_scores = np.sqrt(
    -cross_val_score(
        model_RF,
        X_train,
        y_train,
        cv=5,
        scoring='neg_mean_squared_error'
    )
)

# Cross-validation for MSE
mse_scores = -cross_val_score(
    model_RF,
    X_train,
    y_train,
    cv=5,
    scoring='neg_mean_squared_error'
)

# Train the model on scaled data
model_RF.fit(X_train, y_train)

# Make predictions using the scaled test set
y_pred = model_RF.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print cross-validation scores and evaluation metrics
print("Random Forest Regressor:")
print("Cross-Validation RMSE scores:", rmse_scores)
print("Mean of Cross-Validation RMSE scores:", rmse_scores.mean())
print("Standard Deviation of Cross-Validation RMSE scores:", rmse_scores.std())
print("Cross-Validation MSE scores:", mse_scores)
print("Mean of Cross-Validation MSE scores:", mse_scores.mean())
print("Standard Deviation of Cross-Validation MSE scores:", mse_scores.std())
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared: {r2}")
