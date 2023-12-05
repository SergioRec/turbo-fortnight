"""Initial exploration of the ames dataset.

Metadata: https://www.openintro.org/data/index.php?data=ames
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
obj = df.select_dtypes("object")
obj.nunique(axis=0)

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
# not all categories will be useful, some features very rare
for o in obj:
    count = df.groupby(o)["SalePrice"].count()
    count.plot(kind="bar")
    plt.show()
# %%
