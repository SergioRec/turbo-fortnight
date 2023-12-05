#Step 1
import numpy as np
import pandas as pd

ames_data = pd.read_csv("../data/ames.csv")
zones = ["RL", "RM", "RH"]
ames_data = ames_data[ames_data["MS Zoning"].isin(zones)]
ames_data["Total Bathrooms"] = ames_data["Bsmt Full Bath"] + ames_data["Bsmt Half Bath"] + ames_data["Full Bath"] + ames_data["Half Bath"]
ames_data = ames_data[["Lot Area", "Overall Qual", "Overall Cond", "Gr Liv Area", "TotRms AbvGrd", "Total Bathrooms", "Garage Area", "Bldg Type", "SalePrice"]]
ames_data = ames_data.dropna()
ames_data["Lot Area"] = np.log(ames_data["Lot Area"])
ames_data["Gr Liv Area"] = np.log(ames_data["Gr Liv Area"])
ames_data["SalePrice"] = np.log(ames_data["SalePrice"])
ames_data = ames_data.astype({"Total Bathrooms":"int64"})