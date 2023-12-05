# %%
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# %%
# Step 1
ames_data = pd.read_csv("../data/ames.csv")
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
# splits between train and test datasets
train, test = train_test_split(ames_data, test_size=0.2)

# %%
# list of columns that need scaling
cols = ["Overall Qual", "Overall Cond", "Lot Area", "Gr Liv Area",
        "TotRms AbvGrd", "Total Bsmt SF", "Garage Area", "Total Bathrooms"]

# subset dataset to include target columns + outcome variable
df_sub = train[cols + ["SalePrice"]].copy()


# %%
def scale_data(df, columns):
    """Scale data."""
    scaler = MinMaxScaler().fit(df[columns])
    df[columns] = scaler.transform(df[columns])
    return df


# %%
# scale dataset, but only relevant columns
df_scaled = scale_data(df_sub, cols)

# plot to see distributions after scaling
#for col in cols:
#    sns.displot(data=df_scaled[col])
