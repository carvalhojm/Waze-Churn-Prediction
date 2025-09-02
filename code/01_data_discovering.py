# %%
import numpy as np
import pandas as pd

#%%
df = pd.read_csv("../data/waze_dataset.csv")

# %% 
# Summarizing information
df.head(10)

# %%
df.info()

# %%
df.describe()

# %%
# Investigating null values
# Summarizing statistics of null values
df_null = df[df["label"].isnull()]
df_null.describe()

# %%
# Summarizing statistics of non-null values
df_not_null = df[~df["label"].isnull()]
df_not_null.describe()

# %%
# Investiganting device counts in null values
# counting null values by device
df_null["device"].value_counts()

# %%
# calculating percentages
df_null["device"].value_counts(normalize=True)

# %%
# calculating percentages of non-null values by device
df["device"].value_counts(normalize=True)

# %%
# calculating percentages of churned vs. retained
print(df["label"].value_counts())
print()
print(df["label"].value_counts(normalize=True))

# %%
# calculating median values for churned and retained users
df.groupby("label").median(numeric_only=True)

# %%
# calculating the median kilometers per drive in the last month for retained vs churned users.
df["km_per_drive"] = df["driven_km_drives"] / df["drives"]

df.groupby("label").median(numeric_only=True)[["km_per_drive"]]

# %%
# calculating the median kilometers per driving days in the last month for retained vs churned users.
df["km_per_driving_day"] = df["driven_km_drives"] / df["driving_days"]

df.groupby("label").median(numeric_only=True)[["km_per_driving_day"]]

# %%
# calculating the median drives per driving days in the last month for retained vs churned users.
df["drives_per_driving_day"] = df["drives"] / df["driving_days"]

df.groupby("label").median(numeric_only=True)[["drives_per_driving_day"]]

# %%
# calculating the number os device users per churned vs retained users.
df.groupby(["label","device"]).size()

# %%
# transforming into percentages
df.groupby("label")["device"].value_counts(normalize=True)
