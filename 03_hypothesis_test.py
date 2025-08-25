# %%
import numpy as np
import pandas as pd
from scipy import stats

# %%
df = pd.read_csv("data/waze_dataset.csv")

# %%
df.head()
# %%
# crating a map dictionary
map_dictionary = {"iPhone":2, "Android":1}

# %%
# crating a new device type column
df["device_type"] = df["device"].map(map_dictionary)
df.head()

# %%
df.groupby(["device_type"])["drives"].mean()

# %%
# Hypothesis testing
# isolating "drives" columns for user device type
iphone = df[df["device_type"] == 2]["drives"]
android = df[df["device_type"] == 1]["drives"]

# performing the t-test to find p-value
stats.ttest_ind(a=iphone,b=android, alternative="two-sided", equal_var=False)

# %%
statistic, pvalue = stats.ttest_ind(a=iphone,b=android, alternative="two-sided", equal_var=False)
pvalue_percent = round(pvalue * 100, 2)
print("p-value:",pvalue_percent, "%")