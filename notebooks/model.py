"""Scale and train/test split of the ames dataset.

Metadata: https://jse.amstat.org/v19n3/decock/DataDocumentation.txt

Step 2
"""
# %%
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from pyprojroot import here

# %%
path = here('data/ames.csv')

df = pd.read_csv(path)
df = df[df['MS Zoning'].isin(["RH", "RL", "RP", "RM"])]
df["Lot Area log"] = np.log(df["Lot Area"])
df["Gr Liv Area log"] = np.log(df["Gr Liv Area"])
df["SalePrice log"] = np.log(df["SalePrice"])

# %%
# splits between train and test datasets
train, test = train_test_split(df, test_size=0.2)

# %%
# list of columns that need scaling
cols = ["Overall Qual", "Overall Cond", "Lot Area log", "Gr Liv Area log",
        "TotRms AbvGrd", "Total Bsmt SF", "Garage Area"]

# subset dataset to include target columns + outcome variable
df_sub = train[cols + ["SalePrice log"]].copy()


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
for col in cols:
    sns.displot(data=df_scaled[col])


# %%
