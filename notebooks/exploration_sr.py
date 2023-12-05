"""Initial exploration of the ames dataset.

Metadata: https://jse.amstat.org/v19n3/decock/DataDocumentation.txt
"""
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyprojroot import here

# show all columns
pd.set_option('display.max_columns', None)

# %%
path = here('data/ames.csv')

df = pd.read_csv(path)

# %%
# big differences by neighborhood
sns.boxplot(data=df, x="SalePrice", y="Neighborhood")
plt.show()

# %%
# lot area and price
sns.scatterplot(data=df, x="SalePrice", y="Lot Area")
plt.show()

# %%
# living area and price
sns.scatterplot(data=df, x="SalePrice", y="Gr Liv Area")
plt.show()

# %%
# overall quality/condition and price
sns.barplot(data=df, x="Overall Qual", y="SalePrice", color='lightblue')
plt.show()

sns.barplot(data=df, x="Overall Cond", y="SalePrice", color='lightblue')
plt.show()

# %%
# year sold might have an impact? We might want to remove effect of year
# if trying to predict house prize (as I assume we'd only be interested in
# features of the house itself)
sns.boxplot(x=df["SalePrice"], y=df["Yr Sold"].astype(str))
plt.show()

sns.lineplot(data=df, x='Yr Sold', y='SalePrice')
plt.xticks(df['Yr Sold'].unique())
plt.show()

# %%
# only categorical
obj = df.select_dtypes("object")
obj.nunique(axis=0)

# %%
# not all categories will be useful, some features very rare
for o in obj:
    count = df.groupby(o)["SalePrice"].count()
    count.plot(kind="bar")
    plt.show()

# %%
# only categorical
num = df.select_dtypes("number")

# %%
# num distributions
for n in num:
    s = df[n]
    sns.displot(s)
    plt.show()
# %%
