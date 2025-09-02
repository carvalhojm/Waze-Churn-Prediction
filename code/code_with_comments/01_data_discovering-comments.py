# %%
import numpy as np
import pandas as pd

#%%
df = pd.read_csv("../../data/waze_dataset.csv")

# %% 
# Summarizing information
df.head(10)

# %%
df.info()

# Dataset has 14999 rows and 13 columns, containing int, obj and float.
# "label" has 700 rows missing information

# %%
df.describe()

# Some values indicate anomalies. The maximum "driven_km_drives" is over 21,000 km, which is more than half the circumference of the Earth and may inflate other numbers.

# %%
# Investigating null values
# Summarizing statistics of null values
df_null = df[df["label"].isnull()]
df_null.describe()

# %%
# Summarizing statistics of non-null values
df_not_null = df[~df["label"].isnull()]
df_not_null.describe()

# Analyzing the mean and standard deviations, there does not appear to be a large discrepancy between missing and non-missing records.

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

# The percentage of missing values by each device is consistent with their representation in the data overall.

# %%
# calculating percentages of churned vs. retained
print(df["label"].value_counts())
print()
print(df["label"].value_counts(normalize=True))

# This dataset contains 82% retained users and 18% churned users.

# %%
# calculating median values for churned and retained users
df.groupby("label").median(numeric_only=True)

# The table leads to some insights:
# - Users who churned averaged ~3 more drives in the last month than retained users, but retained users used 
# the app on over twice as many days as churned users in the same time period.
# - The median churned user drove ~200 more kilometers and 2.5 more hours during the last month than the median 
# retained user.
# - It seems that churned users had more drives in fewer days, and their trips were farther and longer in duration.
# - This may suggest a different user profile than the default.
# %%
# calculating the median kilometers per drive in the last month for retained vs churned users.
df["km_per_drive"] = df["driven_km_drives"] / df["drives"]

df.groupby("label").median(numeric_only=True)[["km_per_drive"]]

# The median retained user drove about one more kilometer per drive than the median churned user.

# %%
# calculating the median kilometers per driving days in the last month for retained vs churned users.
df["km_per_driving_day"] = df["driven_km_drives"] / df["driving_days"]

df.groupby("label").median(numeric_only=True)[["km_per_driving_day"]]

# %%
# calculating the median drives per driving days in the last month for retained vs churned users.
df["drives_per_driving_day"] = df["drives"] / df["driving_days"]

df.groupby("label").median(numeric_only=True)[["drives_per_driving_day"]]

# The average churner drove 438 miles per day last month, representing nearly ~240% of the daily distance 
# traveled by retained users. The average churner drove a disproportionate number of trips per day compared 
# to retained users.

# It's clear from these numbers that, regardless of whether the user actually churned, the users represented 
# in this data are serious drivers! It's probably safe to assume that this data isn't representative of 
# typical drivers in general. Perhaps the data—and the churner sample in particular—contains a high proportion 
# of long-haul truck drivers.

# It would be helpful if more data regarding these users' personal information were collected and analyzed 
# to gain a clearer picture of this user group.

# It's also possible that this is why they're leaving the Waze app (which was designed for everyday drivers) 
# once their needs are no longer met.

# %%
# calculating the number os device users per churned vs retained users.
df.groupby(["label","device"]).size()

# %%
# transforming into percentages
df.groupby("label")["device"].value_counts(normalize=True)

# The ratio of iPhone users and Android users is consistent between the churned group and the retained group, and 
# those ratios are both consistent with the ratio found in the overall dataset.

# %%
# GENERAL INSIGHTS
# 1. Only the "label" column, which precisely classifies which users have downloaded the app and which have 
# uninstalled it, is missing. Through analysis, the missing values showed no pattern different from the rest of 
# the data.

# 2. The main benefit is to automatically discard outliers that alter the overall values.

# 3. It would be interesting to have more data containing customer personal information to identify more patterns 
# between active and churned users, such as to confirm the hypothesis that users who churn most often work 
# long-distance driving.

# 4. The percentage of Android users is ~36% and iPhone users ~64%.

# 5.
# - Churned users had an average of 3 more trips than retained users, but retained users used the app practically 
# twice as many days as churned users.
# - The median churned user drove ~200 more kilometers and 2.5 more hours during the last month than the median 
# retained user. - Churned users had more drives in fewer days, and their trips were longer and longer in duration.
# - The average retained user drove about one more kilometer per drive than the average churned user.
# - The average retained user drove about one more kilometer per drive than the average churned user.

# 6. Comparing the percentage of both groups, there was no significant difference in the abandonment rate between 
# iPhone and Android users.